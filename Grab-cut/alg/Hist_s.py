import cv2
import numpy as np
import time

'''
快速直方图模型
'''


class Model:
    # 输入像素集合(L,3)，将三元变量映射到hash值
    def __init__(self, X, colorRanks=(12, 12, 12), component=None):
        X = X / 255  # (79663, 3)
        N = len(X)  # 像素集的总数
        self.colorRank = colorRanks

        self.w = [colorRanks[1] * colorRanks[2], colorRanks[2], 1]  # # [144, 12, 1]
        hash_value = self.rgb2hash(X)  # 将像素集的每个数据映射成哈希值(79663,)

        freq = np.bincount(hash_value, minlength=colorRanks[1] * colorRanks[2] * colorRanks[
            0])  # 拉平并统计，返回长为1728的列表，第i位表示数值i出现的次数。由于像素单一这里形状为(1584,)
        self.hist = freq / N

    # 输入三元像素集，映射成哈希值
    def rgb2hash(self, X):
        idx_img3f_0 = (X[:, 0] * (self.colorRank[0] - 1)).round().astype(np.int32) * self.w[0]
        idx_img3f_1 = (X[:, 1] * (self.colorRank[1] - 1)).round().astype(np.int32) * self.w[1]
        idx_img3f_2 = (X[:, 2] * (self.colorRank[2] - 1)).round().astype(np.int32) * self.w[2]
        idx1i = idx_img3f_0 + idx_img3f_1 + idx_img3f_2
        return idx1i

    # 输入哈希值，映射回三元像素集
    def hash2rgb(self, num_values):
        # 将idx1i映射回RGB值
        total_init0 = (num_values / self.w[0]).astype(np.int32)
        total_init1 = (num_values % self.w[0] / self.w[1]).astype(np.int32)
        total_init2 = (num_values % self.w[1]).astype(np.int32)
        total = np.array([total_init0, total_init1, total_init2]).T  # (630, 3),按频率从小到大
        return total

    # 输入x,返回直方图中对应的概率
    def calc_prob(self, X):
        X = X / 255
        hash_value = self.rgb2hash(X)  # (107753,)
        return self.hist[hash_value]











