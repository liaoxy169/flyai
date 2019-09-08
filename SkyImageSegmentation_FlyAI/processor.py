# -*- coding: utf-8 -*
import numpy as np
from flyai.processor.base import Base
import cv2
from path import DATA_PATH
import os

'''
把样例项目中的processor.py件复制过来替换即可
'''

# 这里处理数据方式仅供参考
img_size = 512

class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, image_path):
        rgb_img_path = os.path.join(DATA_PATH, image_path)
        rgb_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # 获取同样大小的图片
        result_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        h, w = img_size, img_size
        if rgb_img.shape[0] < img_size:
            h = rgb_img.shape[0]
        if rgb_img.shape[1] < img_size:
            w = rgb_img.shape[1]
        result_img[:h, :w, :] = rgb_img[:h, :w, :]

        result_img = result_img.transpose((2, 0, 1))  # channel first
        result_img = result_img / 255.0
        return result_img

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, label_path):
        img_path = os.path.join(DATA_PATH, label_path)
        label = cv2.imread(img_path)

        # 获取同样大小的标签
        result = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        h, w = img_size, img_size
        if label.shape[0] < img_size:
            h = label.shape[0]
        if label.shape[1] < img_size:
            w = label.shape[1]
        result[:h, :w, :] = label[:h, :w, :]

        result = result[:, :, 0]
        result = (result / 255).astype(int)
        return result

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, image_path):
        rgb_img_path = os.path.join(DATA_PATH, image_path)
        rgb_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.transpose((2, 0, 1))  # channel first
        img = rgb_img / 255.0
        return img
    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, prediction):
        prediction = np.squeeze(prediction)
        prediction[prediction>0.5] = 1
        prediction[prediction<=0.5] = 0
        return prediction