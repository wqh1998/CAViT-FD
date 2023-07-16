import torch.nn as nn
import torch
#############################Discriminator#############################

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape  ## input_shape:(3， 256， 256)

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)  ## output_shape = (1, 16, 16)

        def discriminator_block(in_filters, out_filters, normalize=True):  ## 鉴别器块
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, (4, 4), (2, 2), padding=1)]  ## layer += [conv + norm + relu]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),  ## layer += [conv(3, 64) + LeakyReLU]
            *discriminator_block(64, 128),  ## layer += [conv(64, 128) + norm + LeakyReLU]
            *discriminator_block(128, 256),  ## layer += [conv(128, 256) + norm + LeakyReLU]
            *discriminator_block(256, 512),  ## layer += [conv(256, 512) + norm + LeakyReLU]
            nn.ZeroPad2d((1, 0, 1, 0)),  ## layer += [pad]
            nn.Conv2d(512, 1, (4, 4), padding=1)  ## layer += [conv(512, 1)]
        )

    def forward(self, img):  ## 输入(1, 3, 256, 256)
        return self.model(img)  ## 输出(1, 1, 16, 16)

