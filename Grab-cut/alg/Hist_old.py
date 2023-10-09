import cv2
import numpy as np
import time
from operator import itemgetter
'''
快速直方图模型


'''




class Model:
    # 输入像素集合(L,3)，将三元变量映射到hash值
    def __init__(self, X, ratio=0.95, colorRanks=(11,11,11)):
        X = X / 255  # (79663, 3)
        N = len(X)
        # 将归一化矩阵缩的范围，减一个微小值为了最终阶数符合colorRanks（因为有0~colorRanks有colorRanks+1阶）
        self.clrTmp = [colorRanks[0] - 0.0001, colorRanks[1] - 0.0001, colorRanks[2] - 0.0001]  # [11.9999, 11.9999, 11.9999]
        # 哈希映射权值
        self.w = [colorRanks[1] * colorRanks[2], colorRanks[2], 1]  # # [144, 12, 1]
        hash_value = self.rgb2hash(X)  # 将像素集的每个数据映射成哈希值(79663,)
        print('hash_value:', hash_value.shape, hash_value.max(), hash_value.min())
        exit()

        freq = np.bincount(hash_value)  # 拉平并统计，返回长为1728的列表，第i位表示数值i出现的次数。由于像素单一这里形状为(1584,)

        sort_freq = np.sort(freq)  # 频率从小到大排序;(1584,) 17619 0，前面有大量的0，代表这些哈希值没有出现
        sort_pos = np.argsort(freq)  # 排序后每个值对应原来位置的坐标;(1584,) 17619 0
        print('sort_freq:', sort_freq.shape, sort_freq)
        print('sort_pos:', sort_pos.shape, sort_pos)
        # 垂直堆叠两个数组，第一行映射值，第二行原坐标;(2, 1584) 17619 0
        sort_freq_pos = np.vstack((sort_freq, sort_pos))
        print('sort_freq_pos:', sort_freq_pos.shape)
        # 删去频率为0的列;(2, 1, 186) 17619 0,即去掉没出现的值后还剩186个映射值
        sort_freq_pos = sort_freq_pos[:, np.nonzero(sort_freq)]

        num = np.swapaxes(sort_freq_pos, 0, 1)[0]  # 去掉多余维度;(2, 186)
        print('num:', num.shape, '第一行是出现频率，第二行是该映射值原来在freq的坐标')
        print(num[:, :20])
        len_num = maxNum = len(num[0])  # 有效映射值总数;186
        print(f'映射后有{len_num}个有效值')
        maxDropNum = int(np.round(len(X) * (1 - ratio)))  # N*5%，即最多删除maxDropNum个元素
        print(f'最多删除{maxDropNum}个像素')
        accumulate = np.add.accumulate(num[0])  # 对num[0](频数)求前缀和
        cut_pos = np.argwhere(accumulate >= maxDropNum)[0][0]  # 找到前缀和>maxDropNum的第一个位置
        print(f'在num中的第{cut_pos}位划分，前面的为5%，后面的为95%')

        maxNum = maxNum - cut_pos  # 有效值总数-5%分位数=剩下使用映射值数量
        num_values = num[1][::-1]  # 按出现频次从高到低排序，存对应值本来的坐标

        # 最终约简后的元素不能超过256个，不能少于10个
        maxNum = 256 if maxNum > 256 else maxNum
        if maxNum <= 10:
            maxNum = 10 if len(num) > 10 else len(num)

        print(f'最终保留{maxNum}个有效值')
        # 将idx1i映射回RGB值;(186, 3)
        total = self.hash2rgb(num_values)
        print(f'按频率从大到小地排序RGB像素，总共有{len(total)}种像素，后{cut_pos}种像素要用临近像素替代，留下{maxNum}个有效值', total.shape)

        # =========================计算像素距离==============================
        drop = total[:maxNum]  # 前5%种像素
        maxNum2len_Num = total[maxNum:]  # 后95%种像素
        # 距离数组，行代表5%种像素，列代表后95%种像素，遍历计算5%种像素与95%种像素的最近距离
        temp_matrix = np.zeros((len_num - maxNum, maxNum), dtype=np.int32)

        for i, single in enumerate(maxNum2len_Num):  # 5%的像素中的第i中与所有95%的像素求距离
            temp_matrix[i] = np.sum(np.square(single - drop), axis=1)

        arg_min = np.argmin(temp_matrix, axis=1)  # 找距离最小值像素代替原值，第i位代表第i个像素距离最近的是第arg_min[i]个像素;(116,)
        replaceable_colors = num_values[arg_min]  # 通过索引获取5%的颜色值中距离95%的颜色值最近的颜色值;(116,)

        # 有效值：对应坐标
        self.pallet = dict(zip(num_values[:maxNum], range(maxNum)))
        # 遍历待舍弃值和它的代替像素，添加到pallet中
        for num_value, index_dist in zip(num_values[maxNum:], replaceable_colors):
            self.pallet[num_value] = self.pallet[index_dist]  # # 舍弃值：代替像素的坐标

        tmp = hash_value.copy()  # (79663,)
        idx1i_0 = np.zeros(len(X), dtype=np.int32)
        # 遍历每个映射值，第i个映射值
        for i, v in enumerate(tmp):
            idx1i_0[i] = self.pallet[v]  # 映射值v的坐标


        idx1i = idx1i_0  # 每一个像素对应的映射值的坐标

        color3f = np.zeros((1, maxNum, 3), np.float32)  # (1, 70, 3)
        colorNum = np.zeros((1, maxNum), np.int32)  # (1, 70)

        np.add.at(color3f[0], idx1i, X)
        np.add.at(colorNum[0], idx1i, 1)
        colorNum_reshape = colorNum.T
        color3f[0] /= colorNum_reshape

        self.colorNum = (colorNum / N)[0]
        self.hist = np.zeros((colorRanks[0] * colorRanks[1] * colorRanks[2])+1)
        print(self.hist.shape)
        for k, v in self.pallet.items():
            self.hist[k]=self.colorNum[v]

    # 输入三元像素集，映射成哈希值
    def rgb2hash(self, X):
        idx_img3f_0 = (X[:, 0] * self.clrTmp[0]).astype(np.int32) * self.w[0]
        idx_img3f_1 = (X[:, 1] * self.clrTmp[1]).astype(np.int32) * self.w[1]
        idx_img3f_2 = (X[:, 2] * self.clrTmp[2]).astype(np.int32) * self.w[2]
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
        X=X/255
        hash_value = self.rgb2hash(X)  # (107753,)
        return self.hist[hash_value]











