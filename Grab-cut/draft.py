from alg.Quantize import Quantize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
BASE_DIR = os.path.join(BASE_DIR, 'Grab-cut', 'static')
IMG_ROOT = os.path.join(BASE_DIR, 'imgs')
TMP_ROOT = os.path.join(BASE_DIR, 'tmp')
print(IMG_ROOT)

# image = cv2.imread('messi5.jpg')  # 替换为你的图像路径
# # 图像转矩阵并归一化
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_array = image_rgb.astype(np.float32)
# image_array = image_array / 255.0
# # 170 (342, 548) (1, 170, 3) (1, 170)
# binN, idx1i, binColor3f, colorNums1i = Quantize(image_array)
# print('开始输出结果')
# print(binN)   # 170种量化有效像素
# print(idx1i)  # 图中每一位像素对应的colorNums1i中的坐标
# print(binColor3f)  #
# print(colorNums1i)
# print(colorNums1i.sum())
# print(image_rgb.shape[0]*image_rgb.shape[1])








