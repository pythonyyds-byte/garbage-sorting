"""
Created on 2022年6月2日
@author: fjs
@description:本程序用于对垃圾分类原始数据集进行训练集和测试集的划分以及数据分布的统计
@version:1.0
@CopyRight:CQUT
"""

import os,shutil
import random
from pyecharts import options as opts  # pyecharts相关参数
from pyecharts.charts import Bar

train_dataset = '../dataset_train'
test_dataset = '../dataset_test'
train_num = {}  #保存训练集各分类的图片数
test_num = {}   #保存测试集各分类的图片数
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
# garbage_num = [(0, 214), (1, 370), (2, 277), (3, 84), (4, 451), (5, 288), (6, 388),
#                (7, 361), (8, 358), (9, 427), (10, 384), (11, 711), (12, 323), (13, 408),
#                (14, 355), (15, 419), (16, 386), (17, 308), (18, 364), (19, 312), (20, 225),
#                (21, 652), (22, 376), (23, 309), (24, 318), (25, 549), (26, 320), (27, 488),
#                (28, 377), (29, 416), (30, 319), (31, 406), (32, 293), (33, 323), (34, 395),
#                (35, 361), (36, 223), (37, 322), (38, 387), (39, 436)]

garbage_num = {}   #每一类垃圾的数据量字典

for dir_name in os.listdir(train_dataset):
    dir_path = os.path.join(train_dataset,dir_name)
    file_num = 0
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file_name)
        if os.path.isfile(file_path):
            #统计每类垃圾的数据量
            file_num +=1

    garbage_num[int(dir_name)] = file_num

#划分数据集和训练集
for dir_name in os.listdir(train_dataset):
    dir_train_class = os.path.join(train_dataset,dir_name) #训练集分类目录
    dir_test_class = os.path.join(test_dataset,dir_name)  #测试集分类目录
    n = garbage_num[int(dir_name)]  #该分类目录下的图片数
    n1 = int(n/5)  #测试集的数目
    file_name_list = os.listdir(dir_train_class)  #该分类目录下的图片名列表
    for i in range(n1):
        x = random.randint(0,n-i-1)  #随机取图片
        file_name = file_name_list.pop(x)  #获取该随机数下的图片名
        file_path_train = os.path.join(dir_train_class,file_name)  #找到该图片在训练集的路径
        shutil.move(file_path_train,dir_test_class) #将该图片移入测试集对应的分类目录下

#统计训练集的数据量
for dir_name in os.listdir(train_dataset):
    dir_path = os.path.join(train_dataset,dir_name)
    file_num = 0
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file_name)
        if os.path.isfile(file_path):
            #统计每类垃圾的数据量
            file_num +=1

    train_num[int(dir_name)] = file_num
#统计测试集的数据量
for dir_name in os.listdir(test_dataset):
    dir_path = os.path.join(test_dataset,dir_name)
    file_num = 0
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path,file_name)
        if os.path.isfile(file_path):
            #统计每类垃圾的数据量
            file_num +=1

    test_num[int(dir_name)] = file_num

#排序
train_num = sorted(train_num.items())
test_num = sorted(test_num.items())

bar = Bar(init_opts=opts.InitOpts(width='1000px', height='500px'))

x = []
y_train =[]
y_test = []
for i in range(40):
    x.append(label_dict[str(i)])
    y_test.append(test_num[i][1])
    y_train.append(train_num[i][1])


# 设置title
bar.set_global_opts(
    title_opts=opts.TitleOpts(title='训练集和测试集的数据分布'),
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30))  #
)

bar.add_xaxis(xaxis_data=x)
bar.add_yaxis(series_name='train', y_axis=y_train)
bar.add_yaxis(series_name='test', y_axis=y_test)

# 保存
bar.render('./03-训练集和测试集的数据分布.html')