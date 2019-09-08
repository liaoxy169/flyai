# -*- coding: utf-8 -*
import numpy as np
from flyai.processor.base import Base
import cv2
from path import DATA_PATH
import os
'''
把样例项目中的processor.py件复制过来替换即可
'''

class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, rgb_path, thermal_path):
        rgb_img_path = os.path.join(DATA_PATH, rgb_path)
        rgb_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.transpose((2,0,1))
        
        thermal_img_path = os.path.join(DATA_PATH, thermal_path)
        thermal_img = cv2.imread(thermal_img_path,0)
        thermal_img = np.expand_dims(thermal_img,0)
        img = np.concatenate((rgb_img, thermal_img))
        img = img / 255.0
        
        return img

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, mask_path):
        img_path = os.path.join(DATA_PATH, mask_path)
        label = cv2.imread(img_path)
        label = label[:, :, 0]
        _, label = cv2.threshold(label, 150, 255, cv2.THRESH_BINARY)
        label = (label / 255).astype(int)
        return label

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, rgb_path, thermal_path):
        rgb_img_path = os.path.join(DATA_PATH, rgb_path)
        rgb_img = cv2.imread(rgb_img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.transpose((2,0,1))
        
        thermal_img_path = os.path.join(DATA_PATH, thermal_path)
        thermal_img = cv2.imread(thermal_img_path,0)
        thermal_img = np.expand_dims(thermal_img,0)
        img = np.concatenate((rgb_img, thermal_img))
        img = img / 255.0
        return img

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, prediction):
        prediction = np.squeeze(prediction)
        prediction[prediction>0.5] = 1
        prediction[prediction<=0.5] = 0
        return prediction