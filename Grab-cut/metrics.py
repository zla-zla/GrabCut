# 读取图像，调用grabcut获取分割掩码，展示分割效果
import sys
import os
import numpy as np
import math
import cv2 as cv
import igraph as ig
from sklearn.cluster import KMeans
import time
# from alg.grabCut2_clean import *
from alg.grabCut_hist import *
from skimage import io
from sklearn.metrics import recall_score, precision_score, f1_score, jaccard_score
import wandb




def show(img):
    print(img.shape, img.min(), img.max())


def read_mask(image_path):
    # 读取PNG图像
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # (300, 400) 255 0
    return image

def losses(predicted_mask, target_mask):

    # 将预测结果和标签转换为一维数组
    predicted_flat = predicted_mask.flatten()
    target_flat = target_mask.flatten()
    # 计算Recall（召回率）
    recall = recall_score(target_flat, predicted_flat, pos_label=255)

    # 计算Precision（精确率）
    precision = precision_score(target_flat, predicted_flat, pos_label=255)

    # 计算F-measure（F1分数）
    f_measure = f1_score(target_flat, predicted_flat, pos_label=255)

    # 计算Jaccard指数
    jaccard = jaccard_score(target_flat, predicted_flat, pos_label=255)

    return recall, precision, f_measure, jaccard

rect = []
with open('label.txt', 'r') as file:
    for line in file:
        numbers = line.strip().split(' ')
        numbers = [int(num) for num in numbers]
        rect.append(numbers)

dir = "./database/input/"
dir2 = "./database/label/"
x = os.listdir(dir)
label = os.listdir(dir2)
size = len(x)
print(rect)
print(x)
print(label)

# 正常
metrics = np.zeros(4)
for i in range(size):
    print(f'第{i}张图像进行分割', x[i], rect[i])
    img = cv.imread(dir + x[i])
    print('图像大小为：',img.shape)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    T1 = time.time()
    mask = GrabCut(img, mask, rect[i])  # 调用grabcut算法
    print(f'本次分割总共用时为：{round(time.time() - T1, 5)}s')
    mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    area = cv.bitwise_and(img, img, mask=mask)  # mask=mask表示要提取的区域

    cv.imshow("area", area)
    cv.waitKey(0)

    target = read_mask(dir2 + label[i])

    loss = losses(mask, target)
    loss = [round(i, 4) for i in loss]
    print(loss)
    for j in range(4):
        metrics[j] += loss[j]
# 记录第component个成分下指标均值
metrics = metrics/size
print(metrics)

