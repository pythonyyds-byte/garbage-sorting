"""
Created on 2022年6月15日
@author: fjs
@description:本程序用于对
@version:1.0
@CopyRight:CQUT
"""

import torch
from torchvision import datasets,  transforms
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif'] = ['simhei']  # 解决中文

label_dict = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}

TRAIN = '../dataset_train'
TEST = '../dataset_test'

# 数据预处理方法
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放最大边=256
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),  # 归一化[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放最大边=256
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),  # 归一化[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])
# 加载数据库并对其进行预处理
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
test_data = datasets.ImageFolder(TEST, transform=val_transforms)

#DataLoader迭代产生训练数据提供给模型。
batch_size = 32 #批训练数据量的大小

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)#shuffle是否打乱数据
val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)


img, labels = next(iter(train_loader)) #对迭代产生的数据进行获取

classes = [i for i in range(40)]
img = img.permute(0, 2, 3, 1) #转换数据格式
fig = plt.figure(figsize=(25, 8)) #创建背景板

#将图片放到背景板上
for idx in range(batch_size // 4 * 4):
    ax = fig.add_subplot(4, batch_size // 4, idx + 1, xticks=[], yticks=[])
    target_idx = classes[labels[idx]]  # 从标签-->class-->name
    target_name = label_dict[str(target_idx)]
    ax.set_title('{}-{}'.format(target_name, target_idx))
    plt.imshow(img[idx])
plt.show()
