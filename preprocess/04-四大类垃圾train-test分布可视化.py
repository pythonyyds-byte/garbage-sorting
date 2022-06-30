"""
Created on 2022年6月5日
@author: fjs
@description:本程序用于对训练集和测试集的4大类垃圾的数据分布统计
@version:1.0
@CopyRight:CQUT
"""

# 训练数据和验证数据可视化分布,主要是分析每一种标签对应的数量
import os


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
train_dataset = '../dataset_train'
test_dataset = '../dataset_test'
train_num = {}
test_num = {}
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



# 可视化每一个标签对应的label，通过pyecharts绘制图表
from pyecharts import options as opts  # pyecharts相关参数
from pyecharts.charts import Bar

#                 0-5             6-13          14-36        37-39
label_4_name = {0: '其他垃圾', 1: '厨余垃圾', 2: '可回垃圾', 3: '有害垃圾'}
label_4_count_train = {0: 0, 1: 0, 2: 0, 3: 0}
label_4_count_test = {0: 0, 1: 0, 2: 0, 3: 0}
for i in range(40):
    if i <= 5:
        label_4_count_train[0] += train_num[i]
        label_4_count_test[0] += test_num[i]
    elif i > 5 and i <= 13:
        label_4_count_train[1] += train_num[i]
        label_4_count_test[1] += test_num[i]
    elif i > 13 and i <= 36:
        label_4_count_train[2] += train_num[i]
        label_4_count_test[2] += test_num[i]
    else:
        label_4_count_train[3] += train_num[i]
        label_4_count_test[3] += test_num[i]


x = label_4_name.values()
y1 = label_4_count_train.values()
y2 = label_4_count_test.values()
x = list(x)
y1 = list(y1)
y2 = list(y2)

bar = Bar(init_opts=opts.InitOpts(width='1000px', height='500px'))
bar.set_global_opts(
    title_opts=opts.TitleOpts(title='训练集和测试集的4大类数据分布'),
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30))  #
)

bar.add_xaxis(xaxis_data=x)
bar.add_yaxis(series_name='train', y_axis=y1)
bar.add_yaxis(series_name='test', y_axis=y2)

# 保存
bar.render('./04-训练集和测试集的4大类数据分布.html')

