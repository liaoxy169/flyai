# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""
import argparse
import torch
import torch.nn as nn
from flyai.dataset import Dataset
from fcn_segmentation import FCN16s
from torch.optim import Adam, SGD
from model import Model
from path import MODEL_PATH
from flyai.utils.log_helper import train_log

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
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
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
cnn = FCN16s(1).to(device)
optimizer = SGD(cnn.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
criterion = nn.BCELoss()  # 定义损失函数

'''
dataset.get_step() 获取数据的总迭代次数
'''
lowest_loss = 1e5
for i in range(data.get_step()):
    print('----------------'+str(i) + "/" + str(data.get_step())+'-------------------')
    cnn.train()
    x_train, y_train = data.next_train_batch()
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)
    y_train = y_train.unsqueeze(1)
    optimizer.zero_grad()
    outputs = cnn(x_train)
    pred = torch.sigmoid(outputs)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    print("now loss is : %f, lowest loss %f"%(loss.data, lowest_loss))
    # 线上实时打印log
    train_log(train_loss=loss.data.cpu().numpy())
    # 若测试准确率高于当前最高准确率，则保存模型
    if loss.data < lowest_loss:
        lowest_loss = loss.data
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("saved model!!!")