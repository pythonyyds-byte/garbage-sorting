"""
Created on 2022年5月28日
@author: fjs
@description:本程序用于对垃圾分类原始数据集进行分类以及对40种垃圾类别数据分布和宽高比的统计
@version:1.0
@CopyRight:CQUT
"""
import os,shutil
import matplotlib.pyplot as plt
import cv2 as cv
from pyecharts import options as opts
from pyecharts.charts import Bar

original_dataset = '../garbage_classify_v2/train_data_v2'#原始数据目录
train_dataset = '../dataset_train' #分类好的数据集
test_dataset = '../dataset_test'

# 创建每一个垃圾对应的分类文件夹
os.mkdir(train_dataset)
os.mkdir(test_dataset)
for i in range(40):
    class_dir1 = os.path.join(train_dataset, str(i))
    os.mkdir(class_dir1)
    class_dir2 = os.path.join(test_dataset, str(i))
    os.mkdir(class_dir2)



#为一张图片进行类别的分类
for file_name in os.listdir(original_dataset):
    if file_name[-3:] =='txt':
        file_path = os.path.join(original_dataset, file_name)
        text = open(file_path,'r').readline().split(',')
        image_path = os.path.join(original_dataset,text[0].strip())  #原始数据集的图片路径
        new_image_path = os.path.join(train_dataset,text[1].strip()) #训练集的图片路径
        shutil.copy(image_path,new_image_path)  #复制图片

#垃圾类别对应名称表格
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
#统计每一类垃圾的图片张数
garbage_num = {}   #每一类垃圾的数据量字典 格式：{0:214} 类别为0的垃圾对应的图片数
img_radio ={}      #每一个高宽比下的图片数 {1.0:456} 宽高比为1的图片数
for dir_name in os.listdir(train_dataset):
    dir_path = os.path.join(train_dataset,dir_name)
    file_num = 0
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file_name)
        if os.path.isfile(file_path):
            #统计每类垃圾的数据量
            file_num +=1
            # 统计宽高比
            img = cv.imread(file_path)
            x, y = img.shape[:2]
            n = round(x / y, 2)
            if n > 2.5:
                n = round(y / x, 2)
            img_radio[n] = img_radio.get(n, 0) + 1

    garbage_num[int(dir_name)] = file_num

#绘制不同类别下垃圾图片数量的直方图
garbage_num = sorted(garbage_num.items())
x1= []
y1 =[]
for i in garbage_num:
    x1.append(label_dict[str(i[0])])
    y1.append(i[1])
bar = Bar(init_opts=opts.InitOpts(width='1100px', height='500px'))
bar.add_xaxis(xaxis_data=x1)
bar.add_yaxis(series_name='', y_axis=y1)
bar.set_global_opts(
    title_opts=opts.TitleOpts(title='垃圾分类-不同类别的数据分布'),  # 增加标题
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15))  #
)
bar.render(r'.\01-原始数据集的统计.html')

#绘制折线图显示数据的宽高比分布
img_radio = sorted(img_radio.items())
x2 = []
y2 = []
for i in img_radio:
    x2.append(i[0])
    y2.append(i[1])
plt.plot(x2,y2)
plt.title("宽高比统计")
plt.xlabel("比值")
plt.ylabel("图片数")
plt.show()

