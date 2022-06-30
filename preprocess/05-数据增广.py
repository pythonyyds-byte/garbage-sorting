"""
Created on 2022年6月10日
@author: fjs
@description:本程序用于对训练集进行数据增广并对增广后的训练集进行数据分布的统计
@version:1.0
@CopyRight:CQUT
"""
import copy
import cv2
import numpy as np
import os
import random
class DataAugment:
    def __init__(self, debug=False):
        self.debug = debug


    def basic_matrix(self, translation):
        """基础变换矩阵"""
        return np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    def adjust_transform_for_image(self, img, trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix = copy.deepcopy(trans_matrix)
        height, width, channels = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot(
            [self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform(self, img, transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
                                borderValue=0, )  # cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT
        return output

    def apply(self, img, trans_matrix):
        """应用变换"""
        tmp_matrix = self.adjust_transform_for_image(img, trans_matrix)
        out_img = self.apply_transform(img, tmp_matrix)
        if self.debug:
            self.show(out_img)
        return out_img

    def random_vector(self, min, max):
        """生成范围矩阵"""
        min = np.array(min)
        max = np.array(max)
        assert min.shape == max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def show(self, img):
        """可视化"""
        cv2.imshow("outimg", img)
        cv2.waitKey()

    def random_transform(self, img, min_translation, max_translation):
        """平移变换"""
        factor = self.random_vector(min_translation, max_translation)
        trans_matrix = np.array([[1, 0, factor[0]], [0, 1, factor[1]], [0, 0, 1]])
        out_img = self.apply(img, trans_matrix)
        return trans_matrix, out_img

    def random_flip(self, img, factor):
        """水平或垂直翻转"""
        flip_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, flip_matrix)
        return flip_matrix, out_img

    def random_rotate(self, img, factor):
        """随机旋转"""
        angle = np.random.uniform(factor[0], factor[1])
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        out_img = self.apply(img, rotate_matrix)
        return rotate_matrix, out_img

    def random_scale(self, img, min_translation, max_translation):
        """随机缩放"""
        factor = self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, scale_matrix)
        return scale_matrix, out_img

    def random_shear(self, img, factor):
        """随机剪切，包括横向和众向剪切"""
        angle = np.random.uniform(factor[0], factor[1])
        crop_matrix = np.array([[1, factor[0], 0], [factor[1], 1, 0], [0, 0, 1]])
        out_img = self.apply(img, crop_matrix)
        return crop_matrix, out_img


#图像平移
def img_translate(demo,img):
    _, outimg = demo.random_transform(img, (0.1, 0.1), (0.2, 0.2))# (-0.3,-0.3),(0.3,0.3)
    return outimg
#图像垂直变换
def img_vertical(demo,img):
    _, outimg = demo.random_flip(img, (1.0, -1.0))
    return outimg
#图像水平变换
def img_level(demo,img):
    _, outimg = demo.random_flip(img, (-1.0, 1.0))
    return outimg
#图像旋转变换
def img_revolve(demo,img):
    _, outimg = demo.random_rotate(img, (0.5, 0.8))
    return outimg
#图像缩放变换
def img_zoom(demo,img):
    _, outimg = demo.random_scale(img, (1.2, 1.2), (1.3, 1.3))
    return outimg
#图像随机裁剪
def img_crop(demo,img):
    _, outimg = demo.random_shear(img, (0.2, 0.3))
    return outimg
#组合变化
def img_combination(demo,img):
    t1, _ = demo.random_transform(img, (-0.3, -0.3), (0.3, 0.3))
    t2, _ = demo.random_rotate(img, (0.5, 0.8))
    t3, _ = demo.random_scale(img, (1.5, 1.5), (1.7, 1.7))
    tmp = np.linalg.multi_dot([t1, t2, t3])
    outimg = demo.apply(img, tmp)
    return outimg

#统计训练集的数据分布
def get_train_num():
    base_dir = '../dataset_train'  # 分类好的数据集
    garbage_num = {}  # 每一类垃圾的数据量字典
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        file_num = 0
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                # 统计每类垃圾的数据量
                file_num += 1

        garbage_num[int(dir_name)] = file_num
    return sorted(garbage_num.items(), key = lambda x:x[1]),sorted(garbage_num.items())
#获得随机数节选列表
def get_list(num):
    x = []
    for i in range(num):
        x.append(i)
    return x
# [(3, 68), (0, 172), (36, 179), (20, 180), (2, 222), (5, 231), (32, 235), (17, 247),
# (23, 248), (19, 250), (24, 255), (26, 256), (30, 256), (37, 258), (12, 259), (33, 259),
# (14, 284), (8, 287), (35, 289), (7, 289), (18, 292), (1, 296), (22, 301), (28, 302),
# (10, 308), (16, 309), (38, 310), (6, 311), (34, 316), (31, 325), (13, 327), (29, 333),
# (15, 336), (9, 342), (39, 349), (4, 361), (27, 391), (25, 440), (21, 522), (11, 569)]
if __name__ == '__main__':
    garbage_num,garbage_num1 = get_train_num()  #获取训练集的数据分布
    n = garbage_num[-1][1]  # 训练集中最多的一类数据量，所以其他的类的数据量要向着这个数值靠齐
    enhance_num = {} #保存需要增广的垃圾类别和增广的倍数
    for i in garbage_num[:-1]:
        if i[1]>int(n/2):
            x = n-i[1]
        else:
            x = int(n / i[1])-1

        enhance_num[i[0]] = x


    #创建增广文件夹,仅供测试使用，代码测试可行后，数据的增广放到测试集中
    # dir_path_add = '../dataset_add'
    dir_path_train = '../dataset_train'
    # os.mkdir(dir_path_add)
    # for i in range(40):
    #     dir_path_1 = os.path.join(dir_path_add,str(i))
    #     os.mkdir(dir_path_1)

    for i in enhance_num.keys():
        dir_path1 = os.path.join(dir_path_train,str(i))  #需要增广的文件路径
        # dir_path2 = os.path.join(dir_path_add,str(i))  #增广后数据的存放路径
        n = enhance_num[i]
        method = ['img_translate','img_vertical','img_level','img_revolve','img_zoom','img_crop','img_combination']
        demo = DataAugment()
        if n<10: #成倍的进行数据增广
            for file_name in os.listdir(dir_path1):
                file_path1 = os.path.join(dir_path1, file_name)
                img = cv2.imread(file_path1)
                for j in range(n):
                    file_path2 = os.path.join(dir_path1, file_name[:-4] + '_add' + str(j) + '.jpg')
                    img2 = eval(method[j])(demo,img)
                    cv2.imwrite(file_path2,img2)
        else:  #随机部分图片进行增广
            l = get_list(garbage_num1[int(i)][1])
            x = random.sample(l,n)  #挑选的随机数
            file_name = os.listdir(dir_path1) #图片名
            for i in x:
                file_path1 = os.path.join(dir_path1,file_name[i])
                file_path2 = os.path.join(dir_path1, file_name[i] + '_add0' + '.jpg')
                img = cv2.imread(file_path1)
                img2 = eval(method[0])(demo, img)
                cv2.imwrite(file_path2, img2)

    from pyecharts import options as opts
    from pyecharts.charts import Bar
    #重新统计增广后的训练集数据分布
    # 垃圾类别对应名称表格
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
    _,garbage_num=get_train_num()
    x1 = []
    y1 = []
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
    bar.render(r'.\05-增广后训练集的数据分布.html')


