# (1)导入相关的库
# (2)输入参数处理
# (3)数据加载预处理
# (4)工具类:日志,优化器
# (5)模型加载,训练,评估,保存
import os
import torch
import time
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn import metrics  # 计算混淆矩阵
from gcnet.transforms import preprocess
from gcnet.classifier import GarbageClassifier
from gcnet.utils import AverageMeter, save_checkpoint, accuracy
from gcnet.logger import Logger

class_id2name = {
    0: "一次性快餐盒",
    1: "污损塑料",
    2: "烟蒂",
    3: "牙签",
    4: "破碎花盆及碟碗",
    5: "竹筷",
    6: "剩饭剩菜",
    7: "大骨头",
    8: "水果果皮",
    9: "水果果肉",
    10: "茶叶渣",
    11: "菜叶菜根",
    12: "蛋壳",
    13: "鱼骨",
    14: "充电宝",
    15: "包",
    16: "化妆品瓶",
    17: "塑料玩具",
    18: "塑料碗盆",
    19: "塑料衣架",
    20: "快递纸袋",
    21: "插头电线",
    22: "旧衣服",
    23: "易拉罐",
    24: "枕头",
    25: "毛绒玩具",
    26: "洗发水瓶",
    27: "玻璃杯",
    28: "皮鞋",
    29: "砧板",
    30: "纸板箱",
    31: "调料瓶",
    32: "酒瓶",
    33: "金属食品罐",
    34: "锅",
    35: "食用油桶",
    36: "饮料瓶",
    37: "干电池",
    38: "软膏",
    39: "过期药物"
}

def train(args):
    data_path = args.data_path
    save_path = args.save_path
    # (1) load data
    TRAIN = '{}/train'.format(data_path)
    VAL = '{}/val'.format(data_path)
    train_data = datasets.ImageFolder(root=TRAIN, transform=preprocess)
    list1 = train_data.class_to_idx

    val_data = datasets.ImageFolder(root=VAL, transform=preprocess)
    class_list = [class_id2name[i] for i in list(range(len(train_data.class_to_idx.keys())))]
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    # (2) model inital
    GCNet = GarbageClassifier(args.model_name, args.num_classes, args.ngpu, feature_extract=True)

    # (3) Evaluation:Confusion Matrix:Precision  Recall F1-score
    criterion = nn.CrossEntropyLoss()

    # (4) Optimizer
    optimizer = torch.optim.Adam(GCNet.model.parameters(), args.lr)

    # (5) load checkpoint 断点重新加载,制定开始迭代的位置
    epochs = args.epochs
    start_epoch = args.start_epoch

    # (6) model train and val
    best_acc = 0
    if not args.ngpu:
        logger = Logger(os.path.join(save_path, 'log.txt'), title=None)
    else:
        logger = Logger(os.path.join(save_path, 'log_ngpu.txt'), title=None)
    ## 设置logger 的头信息
    logger.set_names(['LR', 'epoch', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    for epoch in range(start_epoch, epochs + 1):
        print('[{}/{}] Training'.format(epoch, args.epochs))
        # train
        train_loss, train_acc = GCNet.train_model(train_loader, criterion, optimizer)
        # val
        test_loss, test_acc = GCNet.test_model(val_loader, criterion, test=None)
        # 核心参数保存logger
        logger.append([args.lr, int(epoch), train_loss, test_loss, train_acc, test_acc])
        print('train_loss:%f, val_loss:%f, train_acc:%f,  val_acc:%f' % (train_loss, test_loss, train_acc, test_acc,))
        # 保存模型
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if not args.ngpu:
            name = 'checkpoint_' + str(epoch) + '.pth.tar'
        else:
            name = 'ngpu_checkpoint_' + str(epoch) + '.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'state_dict': GCNet.model.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()

        }, is_best, checkpoint=save_path, filename=name)
        print('Best acc:')
        print(best_acc)
