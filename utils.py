import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import torch
from torch.autograd import Variable
import warnings

warnings.filterwarnings('ignore')

def getImgInfo(path):
    # path = "f1_2_f2_15_w0.5_0.5.jpg"  ['f1', '2', 'f2', '15', 'w0.5', '0.5.jpg']
    r = path.split("-")
    x1 = r[0]
    x2 = r[2]
    return x1, x2

def gram_matrix(input_tensor):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h) = input_tensor.size()
    features = input_tensor.view(b, c, h)
    features_t = features.transpose(1, 2)  # C和w*h转置
    gram = features.bmm(features_t)# bmm 将features与features_t相乘
    return gram

def ssim_loss(real, fake, num=1000):
    loss = (1 - torch.mean(pytorch_ssim.ssim(real, fake))) * num
    return loss

def symL1(img):
    right = img.clone().detach()[:, :, :, 103:]
    left = img.clone().detach()[:, :, :, 0:103]
    right1 = torch.flip(right, dims=[3])
    x = torch.abs(left - right1)
    return x / 255

def feature_loss(real, fake, num=0.1):
    loss = torch.abs(torch.mean((gram_matrix(real) - gram_matrix(fake))))* num
    return loss

def pixel_loss(real, fake, num=1):
    loss = torch.mean(torch.sum(torch.sum(torch.abs(real - fake), 1), 1)) * num
    return loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

def gradient_penalty(discriminator, batch_x, fake_image):
    # 梯度惩罚项计算函数
    batchsz = batch_x.shape[0]
    # 每个样本均随机采样 t,用于插值
    t = torch.randn([batchsz, 1, 1, 1])
    # 自动扩展为 x 的形状， [b, 1, 1, 1] => [b, h, w, c
    t = torch.broadcast_tensors(t, batch_x.shape)
    # 在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_image
    # 在梯度环境中计算 D 对插值样本的梯度

    interplate = Variable(interplate, requires_grad=True)
    d_interplote_logits = discriminator(interplate)
    d_interplote_logits.backward()
    grads = interplate.grad

    # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
    grads = grads.reshape([grads.shape[0], -1])
    gp = torch.norm(grads, p=1)  # [b]
    # 计算梯度惩罚项
    gp = torch.mean((gp - 1.) ** 2)
    return gp

def d_loss_fn(discriminator, real, fake, d_real_logits, d_fake_logits):
    # 计算 D 的损失函数
    # 计算梯度惩罚项
    gp = gradient_penalty(discriminator, real, fake)
    # WGAN-GP D 损失函数的定义，这里并不是计算交叉熵，而是直接最大化正样本的输出
    # 最小化假样本的输出和梯度惩罚项
    loss = torch.mean(d_fake_logits) - torch.mean(d_real_logits) + 10. * gp

    return loss, gp

def g_loss_fn(discriminator_outputs):
    # 生成器的损失函数
    # WGAN-GP G 损失函数，最大化假样本的输出值
    loss = - torch.mean(discriminator_outputs)
    return loss

# 一致性损失
def identity_loss(real_image, same_image):
    loss = torch.mean(torch.square(real_image - same_image))
    return loss

# 身份损失
def id_loss(fake, tar, ori):
    loss = identity_loss(fake, ori) + torch.div(identity_loss(fake, ori), identity_loss(tar, ori))
    return loss


def sym_loss(input):
    loss = torch.mean(torch.sum(
        torch.sum(symL1(F.avg_pool2d(input, kernel_size=2, strides=2, padding=1)), 1), 1))
    return loss

def reparameterize(mu, log_var):
    # reparameterize 技巧，从正态分布采样 epsion
    eps = torch.randn((16, 16, 16, 512))
    # 计算标准差
    std = torch.exp(log_var) ** 0.5
    # reparameterize 技巧
    z = mu + std * eps
    return z

class Act(nn.Module):
    # 128*128*512
    # F_g,F_l 尺寸相等 都比输出大一圈， F_int通道是他们的一半(512, 512, 256)
    def __init__(self, F_g, F_l, F_int):  # 通道 F_g:大尺寸输入 F_l：前级输入 F_int：他们通道的一半
        super(Act, self).__init__()
        self.W_g = nn.Sequential(  # 步长为1的1*1卷积 BN
            nn.Conv2d(F_g, F_int, (1,1), (1,1), padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )  # 输出：Hg*Wg*F_int

        self.W_x = nn.Sequential(  # 步长为1的1*1卷积 BN
            nn.Conv2d(F_l, F_int, (1,1), (1,1), padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )  # 输出：Hg*Wg*F_int

        self.psi = nn.Sequential(  # 步长为1的1*1卷积 BN
            nn.Conv2d(F_int, 1, (1,1), (1,1), padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g,x 128*128*512
        g1 = self.W_g(g)  # g支路输出
        x1 = self.W_x(x)  # Xl支路输出
        psi = self.relu(g1 + x1)  # 2路信息相加
        psi = self.psi(psi)  # output
        return x * psi  # 与特征图相乘