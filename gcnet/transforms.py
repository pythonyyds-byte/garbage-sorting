# (1) transforms
import io
import torchvision.transforms as transforms
from PIL import Image

# 数据预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  #缩放图片为256X256
    transforms.CenterCrop((224, 224)),  #裁剪中间224X224的区域
    transforms.ToTensor(), #转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #均值和标准差归一化[0,1]
])


def transforms_image(img_path):

    img = Image.open(img_path)
    img = preprocess(img)  # 图片预处理
    img_batch = img.unsqueeze(0)  #增加维度，二维的图片变为三维的张量
    return img_batch
