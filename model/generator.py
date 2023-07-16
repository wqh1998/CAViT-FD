import torch.nn as nn
import torch
from cvitmodel.crossvit import CrossAttentionBlock
from utils import Act
from cvitmodel.selfattention import AttentionBlock

################################残差块儿ResidualBlock######################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(  ## block = [pad + conv + norm + relu + pad + conv + norm]
            nn.ReflectionPad2d(1),  ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, (3, 3)),
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),  ## ReflectionPad2d():利用输入边界的反射来填充输入张量
            nn.Conv2d(in_features, in_features, (3, 3)),
            nn.InstanceNorm2d(in_features),  ## InstanceNorm2d():在图像像素上对HW做归一化，用在风格化迁移
        )

    def forward(self, x):
        return x + self.block(x)  ## 输出为 图像加上网络的残差输出

################################生成器网络GeneratorResNet###################
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=5):  ## (input_shape = (3, 256, 256), num_residual_blocks = 9)
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]  ## 输入通道数channels = 3

        ## 初始化网络结构
        out_features = 64  ## 输出特征数out_features = 64
        self.model1 = nn.Sequential(  ## model = [Pad + Conv + Norm + ReLU]
            nn.ReflectionPad2d(channels),  ## ReflectionPad2d(3):利用输入边界的反射来填充输入张量
            nn.Conv2d(channels, out_features, (7, 7)),  ## Conv2d(3, 64, 7)
            nn.InstanceNorm2d(out_features),  ## InstanceNorm2d(64):在图像像素上对HW做归一化，用在风格化迁移
            nn.ReLU(inplace=True),
        )
        in_features = out_features  ## in_features = 64

        ## 下采样，循环2次
        out_features *= 2  ## out_features = 64 -> 128
        self.model2_1 = nn.Sequential(  ## (Conv + Norm + ReLU) * 2
            nn.Conv2d(in_features, out_features, (3, 3), (2, 2), padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features  ## in_features = 256

        out_features *= 2  ## out_features = 128 -> 256
        self.model2_2 = nn.Sequential(  ## (Conv + Norm + ReLU) * 2
            nn.Conv2d(in_features, out_features, (3, 3), (2, 2), padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features  ## in_features = 256

        model3_1 = [ResidualBlock(out_features)]
        for _ in range(num_residual_blocks):
            model3_1 += [ResidualBlock(out_features)]  ## model += [pad + conv + norm + relu + pad + conv + norm]
        self.model3 = nn.Sequential(*model3_1)

        # 上采样两次
        # for _ in range(2):
        out_features //= 2  ## out_features = 256 -> 128
        self.model4_1 = nn.Sequential(  ## model += [Upsample + conv + norm + relu]
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, (3, 3), (1, 1), padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features  ## out_features = 64

        out_features //= 2  ## out_features = 128 -> 64
        self.model4_2 = nn.Sequential(  ## model += [Upsample + conv + norm + relu]
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, (3, 3), (1, 1), padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

        ##网络输出层
        ##model += [pad + conv + tanh]
        self.model5 = nn.Sequential(nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, (7, 7)), nn.Tanh())  ## 将(3)的数据每一个都映射到[-1, 1]之间

        self.crossattention = CrossAttentionBlock(dim=4096, num_heads=16, mlp_ratio=1., qkv_bias=False,
                                                  qk_scale=None, drop=0.1, attn_drop=0.,
                                             drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True)
        self.selfAttention = AttentionBlock(dim=4096,
                                         num_heads=16,
                                         mlp_ratio=4.,
                                         qkv_bias=False,
                                         qk_scale=None,
                                         drop_ratio=0.,
                                         attn_drop_ratio=0.,
                                         drop_path_ratio=0.,
                                         act_layer=nn.GELU,
                                         norm_layer=nn.LayerNorm)
        self.act1 = Act(64, 64, 256)
        self.act2 = Act(128, 128, 128)

    def forward(self, x, xy):  ## x xy (1, 3, 256, 256)

        x1 = self.model1(x)#x1=(1,64,256,256)
        x2 = self.model2_1(x1)#x2=(1,128,128,128)
        x3 = self.model2_2(x2) #x3=torch.Size([1, 256, 64, 64])
        x4 = self.model3(x3)#x4=([1, 256, 64, 64])
        B, C, H, W = x4.shape
        x5 = x4.view(B, C, H*W)
        x5 = self.selfAttention(x5)#x=torch.Size([1, 256, 4096])

        xy1 = self.model1(xy)#xy1=(1,64,256,256)
        xy2 = self.model2_1(xy1)#xy2=(1,128,128,128)
        xy3 = self.model2_2(xy2)#xy3=torch.Size([1, 256, 64, 64])
        xy4 = self.model3(xy3)#xy4=([1, 256, 64, 64])
        xy5 = xy4.view(B, C, H*W)
        xy5 = self.selfAttention(xy5)
        y2_1 = xy2 - x2
        y2_2 = xy1 - x1
        y6 = xy5 - x5

        y = self.crossattention(x5, xy5)
        y = self.selfAttention(y)
        y = y + y6
        y = y.view(B, C, H, W) #y=(1,256,64,64)
        y1 = self.model3(y)#y1=(1,256,64,64)
        y2 = self.model4_1(y1)#y2=(1,128,128,128)
        y2 = self.act2(y2, y2_1)
        y3 = self.model4_2(y2)#y3=(1,64,256,256)
        y3 = self.act1(y3, y2_2)
        y4 = self.model5(y3)#y3=(1,3,256,256)
        return y4  ##(1, 3, 256, 256)
