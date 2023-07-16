import os
import random
import torch
import datetime
import time
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from datahelper import train_dataloader, dev_dataloader, val_dataloader
from model.generator import GeneratorResNet
# from model.generator_other import Generator
from model.discriminator import Discriminator
from training.lpips import exportPerceptualLoss

############################  datasets  ###################################

## 如果输入的数据集是灰度图像，将图片转化为rgb图像(本次采用的facades不需要这个)
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

############################  models  ###################################
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

#########################################################################
############################  utils  ###################################

## 设置学习率为初始学习率乘以给定lr_lambda函数的值，乘法因子
class Lambdalr:
    def __init__(self, n_epochs, offset, decay_start_epoch):  ## (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  ## 断言，要让n_epochs > decay_start_epoch 才可以
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  ## return    1-max(0, epoch - 3) / (5 - 3)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs * 2 - self.decay_start_epoch)

###########################################################################
############################  cycle_gan  ###################################
## 超参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="mip1_1", help="name of the dataset") ## ../input/facades-dataset
parser.add_argument("--train", type=str, default="train", help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--test", type=str, default="test", help="name of the dataset")  ## ../input/facades-dataset
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--batch_size_test", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=3e-6, help="adam: learning rate") #最好的学习率4e-6/2e-6/1e-7
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
parser.add_argument("--lambda_sym", type=float, default=2.0, help="sym loss weight")
parser.add_argument("--lambda_perceptual", type=float, default=2.0, help="perceptual loss weight")
parser.add_argument("--dataset", type=str, default="mip1", help="choose dataset")#C_3363/F/C_1121/D_1121
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
os.makedirs("images/%s/train" % opt.dataset_name, exist_ok=True)
os.makedirs("images/%s/test" % opt.dataset_name, exist_ok=True)
os.makedirs("save/%s" % opt.dataset_name, exist_ok=True)

## input_shape:(3, 256, 256)
input_shape = (opt.channels, opt.img_height, opt.img_width)

## 创建生成器，判别器对象
generator = GeneratorResNet(input_shape, num_residual_blocks=1)
discriminator = Discriminator(input_shape)
# generator = Generator(img_channels=3, num_features=64, num_residuals=9)
# discriminator = Discriminator(in_channels=3)

## 损失函数
Perceptual = exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=torch.cuda.device_count()).cuda()

## MES 二分类的交叉熵
## L1loss 相比于L2 Loss保边缘
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

## 如果有显卡，都在cuda模式中运行
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# 如果epoch == 0，初始化模型参数;如果epoch == n, 载入训练到第n轮的预训练模型
if opt.epoch != 0:
    # 载入训练到第n轮的预训练模型
    generator = torch.load("save/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch))
    discriminator = torch.load("save/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch))
    print("\nload my model finished !!")
else:
    # 初始化模型参数
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# generator_path_model = "/home/cslg/wqh/2/save/mip1_412_3_QKV/generator_13.pth"
# discriminator_path_model = "/home/cslg/wqh/2/save/mip1_412_3_QKV/discriminator_13.pth"
# generator.load_state_dict(torch.load(generator_path_model))
# discriminator.load_state_dict(torch.load(discriminator_path_model))
## 定义优化函数,优化函数的学习率为0.0003
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 学习率更行进程
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=Lambdalr(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=Lambdalr(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

writer_train = SummaryWriter(log_dir=folder_name_train)
writer_test = SummaryWriter(log_dir=folder_name_test)

def train():
    # ----------
    #  Training
    # ----------
    prev_time = time.time()  ##开始时间
    step1 = 0
    step2 = 0
    for epoch in range(opt.epoch, opt.n_epochs):  ## for epoch in (0, 50)
        for i, batch in enumerate(train_dataloader):  ## batch is a dict, batch['A']:(1, 3, 256, 256), batch['B']:(1, 3, 256, 256)
            step1 += 1
            morph_image = batch['morph_image']
            crimer_image = batch['crimer_image_tensor']
            cocrimer_image = batch['cocrimer_image_tensor']
            ## 读取数据集中的真图片
            ## 将tensor变成Variable放入计算图中，tensor变成variable之后才能进行反向传播求梯度
            real_A = crimer_image.cuda()  ## 真图像B
            real_B = cocrimer_image.cuda()  ## 真图像A
            real_AB = morph_image.cuda()

            ## 全真，全假的标签
            valid = Variable(torch.ones((real_A.size(0), *discriminator.output_shape)), requires_grad=False).cuda()  ## 定义真实的图片label为1 ones((1, 1, 16, 16))
            fake = Variable(torch.zeros((real_A.size(0), *discriminator.output_shape)), requires_grad=False).cuda()  ## 定义假的图片的label为0 zeros((1, 1, 16, 16))

            ## -----------------
            ##  Train Generator
            ## 原理：目的是希望生成的假的图片被判别器判断为真的图片，
            ## 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
            ## 反向传播更新的参数是生成网络里面的参数，
            ## 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
            ## -----------------
            generator.train()
            discriminator.train()
            fake_B = generator(real_A, real_AB)  ## 用真图像A生成的假图像B
            fake_A = generator(real_B, real_AB)  ## 用真图像B生成的假图像A

            #Identity loss ## A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
            loss_id_B = criterion_identity(fake_B, real_B)  ## loss_id_A就是把图像A1放入 B2A 的生成器中，那当然生成图像A2的风格也得是A风格, 要让A1,A2的差距很小
            loss_id_A = criterion_identity(fake_A, real_A)
            loss_identity = (loss_id_A + loss_id_B) / 2 ##Identity loss

            ## GAN loss
            loss_GAN = criterion_GAN(discriminator(fake_B), valid) + criterion_GAN(discriminator(fake_A), valid) ## 用B鉴别器鉴别假图像B，训练生成器的目的就是要让鉴别器以为假的是真的，假的太接近真的让鉴别器分辨不出来

            # Cycle loss 循环一致性损失
            recov_A = generator(fake_B, real_AB)  ## 之前中realA 通过 A -> B 生成的假图像B，再经过 B -> A ，使得fakeB 得到的循环图像recovA，
            loss_cycle_A = criterion_cycle(recov_A, real_A)  ## realA和recovA的差距应该很小，以保证A,B间不仅风格有所变化，而且图片对应的的细节也可以保留
            recov_B = generator(fake_A, real_AB)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            #感知损失
            # 感知损失:LPIPS计算原始图像和重构图像的差异。
            loss_perceptual_A = Perceptual(fake_A, real_A).mean()
            loss_perceptual_B = Perceptual(fake_B, real_B).mean()
            loss_perceptual = (loss_perceptual_A + loss_perceptual_B) / 2
            # loss_perceptual_recov_A = Perceptual(recov_A, real_A).mean()
            # loss_perceptual_recov_B = Perceptual(recov_B, real_B).mean()
            # loss_perceptual = (loss_perceptual_A + loss_perceptual_B + loss_perceptual_recov_A + loss_perceptual_recov_B) / 4


            # Total loss ## 就是上面所有的损失都加起来
            loss_G = loss_GAN + opt.lambda_id * loss_identity+ opt.lambda_cyc * loss_cycle + opt.lambda_perceptual * loss_perceptual
            optimizer_G.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss_G.backward()  ## 将误差反向传播
            optimizer_G.step()  ## 更新参数

            ## -----------------------
            ## Train Discriminator
            ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
            ## -----------------------
            # 真的图像判别为真
            loss_real = criterion_GAN(discriminator(real_B), valid) + criterion_GAN(discriminator(real_A), valid)
            ## 假的图像判别为假(从之前的buffer缓存中随机取一张)
            fake_A = generator(real_B, real_AB)
            fake_B = generator(real_A, real_AB)
            loss_fake = criterion_GAN(discriminator(fake_B.detach()), fake) + criterion_GAN(discriminator(fake_A.detach()), fake)
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            optimizer_D.zero_grad()  ## 在反向传播之前，先将梯度归0
            loss_D.backward()  ## 将误差反向传播
            optimizer_D.step()  ## 更新参数

            writer_train.add_scalar("Loss/discriminator", loss_D, step1)
            writer_train.add_scalar("Loss/generator", loss_G, step1)
            writer_train.add_scalar("Loss/gen(adv)", loss_GAN, step1)
            writer_train.add_scalar("Loss/gen(cycle)", loss_cycle, step1)
            writer_train.add_scalar("Loss/gen(identity)", loss_identity, step1)
            writer_train.add_scalar("Loss/gen(perceptual)", loss_perceptual, step1)

            ## ----------------------
            ##  打印日志Log Progress
            ## ----------------------
            ## 确定剩下的大约时间  假设当前 epoch = 5， i = 100
            batches_done = epoch * len(train_dataloader) + i  ## 已经训练了多长时间 5 * 400 + 100 次
            batches_left = opt.n_epochs * len(train_dataloader) - batches_done  ## 还剩下 50 * 400 - 2100 次
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))  ## 还需要的时间 time_left = 剩下的次数 * 每次的时间
            prev_time = time.time()

            # Print log
            if i % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f, perceptual: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(train_dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        loss_perceptual.item(),
                        time_left,
                    )
                )

            # 每训练40张就保存一组测试集中的图片
            if i % 200 == 0:
                generator.eval()
                fake_B = generator(real_A, real_AB)  ## 用真A生成假B
                fake_A = generator(real_B, real_AB)  ## 用真B生成假A

                # Arange images along x-axis
                ## make_grid():用于把几个图像按照网格排列的方式绘制出来
                real_AB = make_grid(real_AB, nrow=5, normalize=True)
                real_A = make_grid(real_A, nrow=5, normalize=True)
                real_B = make_grid(real_B, nrow=5, normalize=True)
                fake_A = make_grid(fake_A, nrow=5, normalize=True)
                fake_B = make_grid(fake_B, nrow=5, normalize=True)
                # Arange images along y-axis
                ## 把以上图像都拼接起来，保存为一张大图片
                # image_grid = torch.cat((real_A, real_AB, fake_A), dim=1)
                image_grid = torch.cat((real_A, fake_A, real_AB, real_B, fake_B), dim=1)
                save_image(image_grid, "images/%s/train/%s_%d.jpg" % (opt.dataset_name, epoch, i / 200), normalize=False)

        # 更新学习率
        # lr_scheduler_G.step()
        # lr_scheduler_D.step()

####################################################################
                          # test
####################################################################
        if epoch >= 0:
            # 保存测试集溯源出来的共犯图片
            for i, dev_data in enumerate(dev_dataloader, 0):
                step2 += 1
                dev_morph_image = dev_data['morph_image']
                dev_crimer_image = dev_data['crimer_image_tensor']
                dev_cocrimer_image = dev_data['cocrimer_image_tensor']
                real_A = dev_crimer_image.cuda()  ## 真图像B
                real_B = dev_cocrimer_image.cuda()  ## 真图像A
                real_AB = dev_morph_image.cuda()  ## 真图像AB

                # 全真，全假的标签
                valid = Variable(torch.ones((real_A.size(0), *discriminator.output_shape)), requires_grad=False).cuda()  ## 定义真实的图片label为1 ones((1, 1, 16, 16))
                fake = Variable(torch.zeros((real_A.size(0), *discriminator.output_shape)), requires_grad=False).cuda()  ## 定义假的图片的label为0 zeros((1, 1, 16, 16))

                ## -----------------
                ##  Train Generator
                ## 原理：目的是希望生成的假的图片被判别器判断为真的图片，
                ## 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
                ## 反向传播更新的参数是生成网络里面的参数，
                ## 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的, 这样就达到了对抗的目的
                ## -----------------
                fake_B = generator(real_A, real_AB)  ## 用真图像A生成的假图像B
                fake_A = generator(real_B, real_AB)  ## 用真图像B生成的假图像A

                #Identity loss ## A风格的图像 放在 B -> A 生成器中，生成的图像也要是 A风格
                loss_id_B = criterion_identity(fake_B, real_B)  ## loss_id_A就是把图像A1放入 B2A 的生成器中，那当然生成图像A2的风格也得是A风格, 要让A1,A2的差距很小
                loss_id_A = criterion_identity(fake_A, real_A)
                loss_identity = (loss_id_A + loss_id_B) / 2 ## Identity loss

                ## GAN loss
                loss_GAN = criterion_GAN(discriminator(fake_A), valid) ## 用B鉴别器鉴别假图像B，训练生成器的目的就是要让鉴别器以为假的是真的，假的太接近真的让鉴别器分辨不出来

                # Cycle loss 循环一致性损失
                recov_A = generator(fake_B, real_AB)  ## 之前中realA 通过 A -> B 生成的假图像B，再经过 B -> A ，使得fakeB 得到的循环图像recovA，
                loss_cycle_A = criterion_cycle(recov_A, real_A)  ## realA和recovA的差距应该很小，以保证A,B间不仅风格有所变化，而且图片对应的的细节也可以保留
                recov_B = generator(fake_A, real_AB)
                loss_cycle_B = criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # 感知损失
                # 感知损失:LPIPS计算原始图像和重构图像的差异
                loss_perceptual_A = Perceptual(fake_A, real_A).mean()
                loss_perceptual_B = Perceptual(fake_B, real_B).mean()
                loss_perceptual = (loss_perceptual_B + loss_perceptual_A) / 2

                # Total loss ## 就是上面所有的损失都加起来
                loss_G = loss_GAN + opt.lambda_id * loss_identity + opt.lambda_cyc * loss_cycle + opt.lambda_perceptual * loss_perceptual
                ## -----------------------
                ## Train Discriminator
                ## 分为两部分：1、真的图像判别为真；2、假的图像判别为假
                ## -----------------------
                fake_A = generator(real_B, real_AB)
                fake_B = generator(real_A, real_AB)
                # 真的图像判别为真
                loss_real = (criterion_GAN(discriminator(real_B), valid) + criterion_GAN(discriminator(real_A), valid)) / 2
                loss_fake = criterion_GAN(discriminator(fake_A.detach()), fake)
                # Total loss
                loss_D = (loss_real + loss_fake) / 2

                writer_test.add_scalar("Loss/discriminator", loss_D, step2)
                writer_test.add_scalar("Loss/generator", loss_G, step2)
                writer_test.add_scalar("Loss/gen(adv)", loss_GAN, step2)
                writer_test.add_scalar("Loss/gen(cycle)", loss_cycle, step2)
                writer_test.add_scalar("Loss/gen(identity)", loss_identity, step2)
                writer_test.add_scalar("Loss/gen(perceptual)", loss_perceptual, step2)

                fake_B = fake_B.detach().cpu()
                fake_A = fake_A.detach().cpu()
                real_A = real_A.detach().cpu()
                real_B = real_B.detach().cpu()
                real_AB = real_AB.detach().cpu()

                if i % 10 == 0:
                    print(
                        "[test][D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f, perceptual: %f]"
                        % (
                            loss_D.item(),
                            loss_G.item(),
                            loss_GAN.item(),
                            loss_cycle.item(),
                            loss_identity.item(),
                            loss_perceptual.item(),
                        )
                    )

                if i % 20 == 0:

                    real_A = make_grid(real_A, nrow=5, normalize=True)
                    real_B = make_grid(real_B, nrow=5, normalize=True)
                    fake_A = make_grid(fake_A, nrow=5, normalize=True)
                    fake_B = make_grid(fake_B, nrow=5, normalize=True)
                    real_AB = make_grid(real_AB, nrow=5, normalize=True)

                    img = torch.cat([real_A, fake_A, real_AB, real_B, fake_B], dim=1)
                    save_image(img, "images/%s/test/%d_%d.jpg" % (opt.dataset_name, epoch, i / 20), normalize=False)

            os.makedirs("photo/%s/%d/fake_A" % (opt.dataset_name, epoch), exist_ok=True)
            os.makedirs("photo/%s/%d/fake_B" % (opt.dataset_name, epoch), exist_ok=True)
            os.makedirs("photo/%s/%d/real" % (opt.dataset_name, epoch), exist_ok=True)
            os.makedirs("val_images/%s/%d" % (opt.dataset_name, epoch), exist_ok=True)

            for i, val_data in enumerate(val_dataloader, 0):
                val_morph_image = val_data['morph_image']
                val_crimer_image = val_data['crimer_image_tensor']
                val_cocrimer_image = val_data['cocrimer_image_tensor']
                real_A = val_crimer_image.cuda()  ## 真图像A
                real_B = val_cocrimer_image.cuda()  ## 真图像B
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
                save_image(fake_A, "photo/%s/%d/fake_A/%d.jpg" % (opt.dataset_name, epoch, i), normalize=True)
                save_image(fake_B, "photo/%s/%d/fake_B/%d.jpg" % (opt.dataset_name,epoch, i), normalize=True)
                save_image(real_A, "photo/%s/%d/real/%d_A.jpg" % (opt.dataset_name,epoch ,i), normalize=True)
                save_image(real_B, "photo/%s/%d/real/%d_B.jpg" % (opt.dataset_name,epoch, i), normalize=True)
                save_image(real_AB, "photo/%s/%d/real/%d_AB.jpg" % (opt.dataset_name,epoch, i), normalize=True)

                real_AB = make_grid(real_AB, nrow=5, normalize=True)
                real_A = make_grid(real_A, nrow=5, normalize=True)
                real_B = make_grid(real_B, nrow=5, normalize=True)
                fake_B = make_grid(fake_B, nrow=5, normalize=True)
                fake_A = make_grid(fake_A, nrow=5, normalize=True)
                img = torch.cat([real_A, fake_A, real_AB, real_B, fake_B], dim=1)
                save_image(img, "val_images/%s/%d/%d.jpg" % (opt.dataset_name, epoch, i), normalize=False)

        ## 训练结束后，保存模型
        torch.save(generator.state_dict(), "save/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "save/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
        print("\nsave my model finished !!")

writer_train.close()
writer_test.close()

## 函数的起始
if __name__ == '__main__':
    train()


