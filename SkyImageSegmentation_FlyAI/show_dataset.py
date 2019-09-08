from flyai.dataset import Dataset
import cv2
import os
import pandas as pd
import numpy as np

base_path = 'data/input'
df = pd.read_csv(os.path.join(base_path, 'dev.csv'), sep=',', header=0)
df['image_path'] = df['image_path'].map(lambda x: os.path.join(base_path, x))
df['label_path'] = df['label_path'].map(lambda x: os.path.join(base_path, x))

img_size = 512
for i in range(len(df)):
    print('---------%d img ----------'%i)
    line = df.iloc[i]
    img = cv2.imread(line['image_path'])
    # 截取同样大小
    result_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = img_size, img_size
    if img.shape[0] < img_size:
        h = img.shape[0]
    if img.shape[1] < img_size:
        w = img.shape[1]
    result_img[:h, :w, :] = img[:h, :w, :]
    print(img.shape)

    label = cv2.imread(line['label_path'])
    label = (label * 255).astype(np.uint8)
    # 截取同样大小
    result = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = img_size, img_size
    if label.shape[0] < img_size:
        h = label.shape[0]
    if label.shape[1] < img_size:
        w = label.shape[1]
    result[:h, :w, :] = label[:h, :w, :]
    cv2.imshow('result_img', result_img)
    cv2.imshow('result', result)
    cv2.waitKey()

