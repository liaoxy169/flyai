# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import argparse
import torch
import torch.nn as nn
from net import Net
from flyai.dataset import Dataset
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
from model import Model
from path import MODEL_PATH


'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
data = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(data)

print('batch size: %d, epoch size： %d'%(args.BATCH, args.EPOCHS))

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device) 

'''
实现自己的网络结构
'''
cnn = Net(num_classes=1 ,num_channels=4).to(device)
optimizer = Adam(cnn.parameters(), lr=0.005, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.BCELoss()  # 定义损失函数

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def calc_loss(pred, target, bce_weight=0.2):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    print(torch.cat((pred.sum(2).sum(2),target.sum(2).sum(2)),1))
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss
'''
dataset.get_step() 获取数据的总迭代次数
'''
lowest_loss = 1e5
for i in range(data.get_step()):
    cnn.train()
    x_train, y_train = data.next_train_batch() 
    x_test, y_test = data.next_validation_batch()

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)
    y_train = y_train.unsqueeze(1)
    outputs = cnn(x_train)
    optimizer.zero_grad()
    loss = calc_loss(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(loss)
    # 若测试准确率高于当前最高准确率，则保存模型
    if loss.data < lowest_loss:
        lowest_loss = loss.data
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, lowest loss %g" % (i, lowest_loss))
    print(str(i) + "/" + str(data.get_step()))
