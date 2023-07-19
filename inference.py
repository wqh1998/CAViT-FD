import os
import torch
import argparse
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from datahelper_test import val_dataloader
from model.generator import GeneratorResNet
from model.discriminator import Discriminator

# 定义参数初始化函数
def weights_init_normal(m):
    classname = m.__class__.__name__  ## m作为一个形参，原则上可以传递很多的内容, 为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字.
    if classname.find("Conv") != -1:  ## find():实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)  ## m.weight.data表示需要初始化的权重。nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        if hasattr(m, "bias") and m.bias is not None:  ## hasattr():用于判断ma是否包含对应的属性bias, 以及bias属性是否不为空.
            torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_():表示将偏差定义为常量0.
    elif classname.find("BatchNorm2d") != -1:  ## find():实现查找classname中是否含有BatchNorm2d字符，没有返回-1；有返回0.
        torch.nn.init.normal_(m.weight.data, 1.0,0.02)  ## m.weight.data表示需要初始化的权重. nn.init.normal_():表示随机初始化采用正态分布，均值为0，标准差为0.02.
        torch.nn.init.constant_(m.bias.data, 0.0)  ## nn.init.constant_():表示将偏差定义为常量0.

## 设置学习率为初始学习率乘以给定lr_lambda函数的值，乘法因子
class Lambdalr:
    def __init__(self, n_epochs, offset, decay_start_epoch):  ## (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  ## 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  ## return    1-max(0, epoch - 3) / (5 - 3)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs * 2 - self.decay_start_epoch)

#########################################################################################################
#############################################  CAViT-FD  ################################################
## 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=0, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="neu_0.1~0.9_9pth", help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--train", type=str, default="train", help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--test", type=str, default="test", help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--batch_size_test", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-6, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=3, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=8, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=6.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=10.0, help="identity loss weight")
parser.add_argument("--lambda_sym", type=float, default=6.0, help="sym loss weight")
parser.add_argument("--lambda_perceptual", type=float, default=2.0, help="perceptual loss weight")
parser.add_argument("--dataset", type=str, default="D_1121", help="choose dataset")#C_3363/F/C_1121/D_1121
opt = parser.parse_args()

folder_name_train = f'runs/{opt.dataset_name}/train/lr-{opt.lr}_\
n_epochs-{opt.n_epochs}_\
batch_size-{opt.batch_size}_\
lambda_cyc-{opt.lambda_cyc}_\
lambda_id-{opt.lambda_id}_\
lambda_sym-{opt.lambda_sym}_\
lambda_perceptual-{opt.lambda_perceptual}_\
dataset-{opt.dataset}'

folder_name_test = f'runs/{opt.dataset_name}/test/lr-{opt.lr}_\
n_epochs-{opt.n_epochs}_\
batch_size-{opt.batch_size_test}_\
lambda_cyc-{opt.lambda_cyc}_\
lambda_id-{opt.lambda_id}_\
lambda_sym-{opt.lambda_sym}_\
lambda_perceptual-{opt.lambda_perceptual}_\
dataset-{opt.dataset}'

## 创建文件夹
# os.makedirs("images/%s/train" % opt.dataset_name, exist_ok=True)
os.makedirs("photo/%s/fake_A" % opt.dataset_name, exist_ok=True)
os.makedirs("photo/%s/fake_B" % opt.dataset_name, exist_ok=True)
os.makedirs("photo/%s/real" % opt.dataset_name, exist_ok=True)
os.makedirs("val_images/%s"% opt.dataset_name, exist_ok=True)

## input_shape:(3, 256, 256)
input_shape = (opt.channels, opt.img_height, opt.img_width)

# if opt.epoch == 0:
#     generator = torch.load("save/226_4/generator_0.pth")
#     discriminator = torch.load("save/226_4/discriminator_0.pth")
#     print("\nload my model finished !!")

print("model loading")
generator_path_model = "./generator.pth"
discriminator_path_model = "./discriminator.pth"
## 创建生成器，判别器对象
generator = GeneratorResNet(input_shape, num_residual_blocks=1)
discriminator = Discriminator(input_shape)
generator.load_state_dict(torch.load(generator_path_model))
discriminator.load_state_dict(torch.load(discriminator_path_model))
print("finish")

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()

####################################################################
                          # test
####################################################################
def test():
    # 保存测试集溯源出来的共犯图片
    for i, val_data in enumerate(val_dataloader, 0):
        val_morph_image = val_data['morph_image']
        val_crimer_image = val_data['crimer_image_tensor']
        val_cocrimer_image = val_data['cocrimer_image_tensor']
        real_A = val_crimer_image.cuda()  ## 真图像B
        real_B = val_cocrimer_image.cuda()  ## 真图像A
        real_AB = val_morph_image.cuda()  ## 真图像AB

        fake_B = generator(real_A, real_AB)  ## 用真图像A生成的假图像B
        fake_A = generator(real_B, real_AB)  ## 用真图像B生成的假图像A

        fake_B = fake_B.detach().cpu()
        fake_A = fake_A.detach().cpu()
        real_A = real_A.detach().cpu()
        real_B = real_B.detach().cpu()
        real_AB = real_AB.detach().cpu()

        print("[test][i: %d]" % i)

        # 超分
        save_image(fake_A, "photo/%s/fake_A/%d.jpg" % (opt.dataset_name, i), normalize=True)
        save_image(fake_B, "photo/%s/fake_B/%d.jpg" % (opt.dataset_name, i), normalize=True)
        save_image(real_A, "photo/%s/real/%d_A.jpg" % (opt.dataset_name, i), normalize=True)
        save_image(real_B, "photo/%s/real/%d_B.jpg" % (opt.dataset_name, i), normalize=True)
        save_image(real_AB, "photo/%s/real/%d_AB.jpg" % (opt.dataset_name, i), normalize=True)

        real_AB = make_grid(real_AB, nrow=5, normalize=True)
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        img = torch.cat([real_A, fake_A, real_AB, real_B, fake_B], dim=1)
        save_image(img, "val_images/%s/%d.jpg" % (opt.dataset_name, i), normalize=False)

## 函数的起始
if __name__ == '__main__':
    test()

