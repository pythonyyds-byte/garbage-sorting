
"""
Created on 2022年6月12日
@author: fjs
@description:本程序用于对图片进行数据增强，实现图片的缩放，中心裁剪以及归一化
@version:1.0
@CopyRight:CQUT
"""

# (1)裁剪-Crop
# 中心裁剪 . CenterCrop
# 随机裁剪 .RandomCrop
# 随机长宽比裁剪： RandomResizeCrop
# 上下左右中心裁剪： FiveCrop
# 上下左右中心裁剪后翻转：.TenCrop

# (2) 翻转和旋转 --Flip and Rotation
# 依概率p水平翻转 RandomHorizontalFlip(p=0.5)
# 依概率p垂直翻转 RandomVerticalFlip(p=0.5)
# 随机翻转：.RandomRotation()

# (3) 图像变换
# 缩放 transforms.Resize()
# 标准化：transforms.Normalize()
# 转为tensor ，并归一化到[0,1]  .ToTensor  填充 .Pad
# 亮度  ，对比度，饱和度，.ColorJitter()
# 灰度转换 .Grayscale()
# 线性变换 .LinearTransformation()
# 放射变换 .RandomAffine()
# 概率p转换为灰度图： .RandomGrayScale()
# 将数据转换为PILImage .ToPILImage()

from torchvision import transforms
from PIL import Image

img_path = '../dataset_train/0/img_1.jpg'
img = Image.open(img_path)
import matplotlib.pyplot as plt

ax = plt.imshow(img)
plt.show()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放最大边=256
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),  # 归一化[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
res_img = preprocess(img)
res = res_img.permute(1, 2, 0)  # Tensor中 c，h，w，转换数据格式
ax = plt.imshow(res)  # PIL 中h,w,c
plt.show()
print(res_img.size)
