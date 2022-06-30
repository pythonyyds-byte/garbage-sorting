"""
Created on 2022年6月1日
@author: fjs
@description:本程序用于对垃圾分类原始数据集的4大类垃圾进行数据分布统计
@version:1.0
@CopyRight:CQUT
"""

import os
train_dataset = '../dataset_train'
#以下为01中统计的40小类垃圾的数据分布
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

garbage_num = sorted(garbage_num.items())
# 可视化每一个标签对应的label，通过pyecharts绘制图表
from pyecharts import options as opts  # pyecharts相关参数
from pyecharts.charts import Bar

#                 0-5             6-13          14-36        37-39
label_4_name = {0: '其他垃圾', 1: '厨余垃圾', 2: '可回垃圾', 3: '有害垃圾'}
label_4_count = {0: 0, 1: 0, 2: 0, 3: 0}

for i in range(40):
    if i <= 5:
        label_4_count[0] += garbage_num[i][1]
    elif i > 5 and i <= 13:
        label_4_count[1] += garbage_num[i][1]
    elif i > 13 and i <= 36:
        label_4_count[2] += garbage_num[i][1]
    else:
        label_4_count[3] += garbage_num[i][1]


x = label_4_name.values()
y = label_4_count.values()
x = list(x)
y = list(y)
bar = Bar(init_opts=opts.InitOpts(width='1100px', height='500px'))
bar.add_xaxis(xaxis_data=x)
bar.add_yaxis(series_name='', y_axis=y)
# 设置全局变量
bar.set_global_opts(
    title_opts=opts.TitleOpts(title='垃圾分类-不同类别的数据分布'),  # 增加标题
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15))  #
)
bar.render('./02-四大类垃圾的数据分布.html')

