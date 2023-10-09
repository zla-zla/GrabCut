# 读取图像，调用grabcut获取分割掩码，展示分割效果
import sys
import os
import numpy as np
import math
import cv2 as cv
import igraph as ig
from sklearn.cluster import KMeans
import time
from alg.grabCut_GMM import *
# from alg.grabCut_hist import *


# 读取图像
filename = 'messi5.jpg'
img = cv.imread(filename)

# 准备掩码(H,W)和输出(H,W,3)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
# 准备矩形框
rect = (69, 59, 389, 277)  # 左下角坐标+宽和高
T1 = time.time()
mask = GrabCut(img, mask, rect)  # 调用grabcut算法
print(f'本次分割总共用时为：{round(time.time()-T1,5)}s')

mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
area = cv.bitwise_and(img, img, mask=mask2)  # mask=mask表示要提取的区域
cv.imshow("area", area)
cv.waitKey(0)
