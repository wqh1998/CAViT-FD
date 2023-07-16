import torch
from utils import getImgInfo
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

# 读取文件路径
train_morph_path = "data_mip1/train/morph/"
train_criminor_path = "mip/data_mip1/train/real/"
train_cocriminor_path = "mip/data_mip1/train/real/"
train_files = os.listdir(train_morph_path)

dev_morph_path = "mip/data_mip1/test/morph/"
dev_criminor_path = "mip/data_mip1/test/real/"
dev_files = os.listdir(dev_morph_path)

# LAMBDA = 10
im_names = []
im1_names = []
dev_names = []
# dev1_names = []
image_size = 256
real1_list = []
real2_list = []
dev_real1_list = []
dev_real2_list = []

val_names = []
# val1_names = []
val_real1_list = []
val_real2_list = []

# 读取训练集融合人脸图片名称
for file_name in train_files:
    im_names.append(file_name)
    # for i in range(5):
    #     im_names.append(file_name)

for j in range(len(im_names)):
    # 得到罪犯x1和共犯x2的name
    x1, x2 = getImgInfo(im_names[j])
    real1_list.append(x1 + '.png')
    real2_list.append(x2 + '.png')
    # for i in range(1, 6):
    #     real1_list.append(x1 + '_' + str(i) + '.jpg')
    #     real2_list.append(x2 + '_' + str(i) + '.jpg')

# 读取验证集融合人脸图片名称
for file_name in dev_files:
    dev_names.append(file_name)

for k in range(len(dev_names)):
    # crimer_image_tensors = (128, 128, 3)
    # cocrimer_image_tensors = (128, 128, 3)
    # 得到罪犯x1和共犯x2的name
    x1, x2 = getImgInfo(dev_names[k])
    dev_real1_list.append(x1 + '.png')
    dev_real2_list.append(x2 + '.png')

    # dev_real1_list.append(x1 + '_1' + '.jpg')
    # dev_real2_list.append(x2 + '_1' + '.jpg')

transform = transforms.Compose([transforms.Resize(int(image_size * 1.05)),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class trainDataset(Dataset):
    def __init__(self, train_cocriminor, train_criminor, train_morph, transform, target_transform=None):

        self.train_cocriminor = train_cocriminor
        self.train_criminor = train_criminor
        self.train_morph = train_morph
        self.template = train_cocriminor
        self.transforms = transform
        self.target_transform = target_transform

    def __len__(self): #1529 1146 1121 3363 5605 22050 1125 5625 796 668
        length = 1121
        return length

    def __getitem__(self, idx):
        morph_image_path = Image.open(train_morph_path + im_names[idx])
        morph_image = self.transforms(morph_image_path).to(device)

        crimer_image = Image.open(self.train_criminor + real1_list[idx])
        crimer_image_tensor = self.transforms(crimer_image).to(device)

        cocrimer_image = Image.open(self.train_cocriminor + real2_list[idx])
        cocrimer_image_tensor = self.transforms(cocrimer_image).to(device)

        template = cocrimer_image_tensor
        data = dict(
            morph_image=morph_image,
            crimer_image_tensor=crimer_image_tensor,
            cocrimer_image_tensor=cocrimer_image_tensor,
            template=template,
            morph_name=im_names[idx],
            cocriminor_name=real2_list[idx],
            criminor_name=real1_list[idx]
        )
        return data

class testDataset(Dataset):
    def __init__(self, dev_cocriminor, dev_criminor, dev_morph, transform, target_transform=None):
        self.dev_cocriminor = dev_cocriminor
        self.dev_criminor = dev_criminor
        self.dev_morph = dev_morph
        self.template = dev_cocriminor
        self.transforms = transform
        self.target_transform = target_transform

    def __len__(self):#646 296 888 1495 571 197 203
        length = 169
        return length

    def __getitem__(self, idx):
        dev_image_path = Image.open(dev_morph_path + dev_names[idx])
        dev_morph_image = self.transforms(dev_image_path).to(device)

        dev_crimer_image = Image.open(self.dev_criminor + dev_real1_list[idx])
        dev_crimer_image_tensor = self.transforms(dev_crimer_image).to(device)

        dev_cocrimer_image = Image.open(self.dev_cocriminor + dev_real2_list[idx])
        dev_cocrimer_image_tensor = self.transforms(dev_cocrimer_image).to(device)

        template = dev_cocrimer_image_tensor
        dev_data = dict(
            morph_image=dev_morph_image,
            crimer_image_tensor=dev_crimer_image_tensor,
            cocrimer_image_tensor=dev_cocrimer_image_tensor,
            template=template,
            morph_name=dev_names[idx],
            cocriminor_name=dev_real2_list[idx],
            criminor_name=dev_real1_list[idx]
        )
        return dev_data

traindataset = trainDataset(train_cocriminor=train_cocriminor_path, train_criminor=train_criminor_path, train_morph= train_morph_path, transform=transform)
train_dataloader = DataLoader(traindataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

dev_dataset = testDataset(dev_cocriminor=dev_criminor_path, dev_criminor=dev_criminor_path, dev_morph=dev_morph_path, transform=transform)
dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

val_morph_path = "/home/cslg/wqh/mip/data_mip1/test/morph/"
val_criminor_path = "/home/cslg/wqh/mip/data_mip1/test/real/"
val_files = os.listdir(val_morph_path)

for file_name in val_files:
    val_names.append(file_name)

for q in range(len(val_names)):
    # crimer_image_tensors = (128, 128, 3)
    # cocrimer_image_tensors = (128, 128, 3)
    # 得到罪犯x1和共犯x2的name
    x1, x2 = getImgInfo(val_names[q])
    val_real1_list.append(x1 + '.png')
    val_real2_list.append(x2 + '.png')

    # val_real1_list.append(x1 + '_1' + '.jpg')
    # val_real2_list.append(x2 + '_1' + '.jpg')

class valDataset(Dataset):
    def __init__(self, val_cocriminor, val_criminor, val_morph, transform, target_transform=None):
        self.val_cocriminor = val_cocriminor
        self.val_criminor = val_criminor
        self.val_morph = val_morph
        self.template = val_cocriminor
        self.transforms = transform
        self.target_transform = target_transform

    def __len__(self):#646 296 888 570 203 169
        length = 169
        return length

    def __getitem__(self, idx):
        val_image_path = Image.open(val_morph_path + val_names[idx])
        val_morph_image = self.transforms(val_image_path).to(device)

        val_crimer_image = Image.open(self.val_criminor + val_real1_list[idx])
        val_crimer_image_tensor = self.transforms(val_crimer_image).to(device)

        val_cocrimer_image = Image.open(self.val_cocriminor + val_real2_list[idx])
        val_cocrimer_image_tensor = self.transforms(val_cocrimer_image).to(device)

        template = val_cocrimer_image_tensor
        val_data = dict(
            morph_image=val_morph_image,
            crimer_image_tensor=val_crimer_image_tensor,
            cocrimer_image_tensor=val_cocrimer_image_tensor,
            template=template,
            morph_name=val_names[idx],
            cocriminor_name=val_real2_list[idx],
            criminor_name=val_real1_list[idx]
        )
        return val_data

val_dataset = valDataset(val_cocriminor=val_criminor_path, val_criminor=val_criminor_path, val_morph=val_morph_path, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
