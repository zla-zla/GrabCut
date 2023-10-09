import cv2
import numpy as np
import time

def show(img):
    print(img.shape,img.max(),img.min())

# 图像归一化矩阵，保留率，量化阶数
def Quantize(img3f,ratio=0.95,colorNums=(12,12,12)):
    show(img3f)  # (H, W, 3)
    # 将归一化矩阵缩的范围，减一个微小值为了最终阶数符合colorNums（因为有0~colorNums有colorNums+1阶）
    clrTmp = [colorNums[0]-0.0001, colorNums[1]-0.0001, colorNums[2]-0.0001]  # [11.9999, 11.9999, 11.9999]
    # 哈希映射权值
    w = [colorNums[1] * colorNums[2], colorNums[2], 1]  # # [144, 12, 1]

    height, width = img3f.shape[:2]
    img3f_0, img3f_1, img3f_2 = cv2.split(img3f)  # 拆分R,G,B三个分量
    # 将三个分量通过哈希映射映射到一个变量idx1i上
    idx_img3f_0 = (img3f_0 * clrTmp[0]).astype(np.int32)* w[0]
    idx_img3f_1 = (img3f_1 * clrTmp[1]).astype(np.int32)* w[1]
    idx_img3f_2 = (img3f_2 * clrTmp[2]).astype(np.int32)* w[2]
    idx1i = idx_img3f_0 + idx_img3f_1 + idx_img3f_2  # (H, W)，每个像素映射后的值

    #==================== 统计像素出现频数 ======================
    bincount_pallet = np.bincount(idx1i.reshape(1,-1)[0])  # 拉平并统计，返回长为1728的列表，第i位表示数值i出现的次数
    print('bincount_pallet:', len(bincount_pallet), bincount_pallet.shape)
    sort_pallet = np.sort(bincount_pallet)  # 映射值从小到大排序;(1728,) 17619 0
    argsort_pallet = np.argsort(bincount_pallet)  # 排序后每个值对应原来位置的坐标;(1728,) 17619 0
    # 垂直堆叠两个数组，第一行映射值，第二行原坐标;(2, 1728) 17619 0
    numpy_pallet = np.vstack((sort_pallet, argsort_pallet))
    # 取两行和非零列;(2, 1, 630) 17619 0,即去掉没出现的值后还剩630个映射值
    numpy_pallet = numpy_pallet[:, np.nonzero(sort_pallet)]
    print(numpy_pallet)
    num = np.swapaxes(numpy_pallet, 0, 1)[0]                      # 维度交换，其实是去掉多余维度;(2, 630)
    len_num = maxNum = len(num[0])                                # 有效映射值总数;630

    maxDropNum = int(np.round(height * width * (1 - ratio)))      # N*5%，即最多删除maxDropNum个元素
    sum_pallet = np.add.accumulate(num[0])                        # 对num[0](频数)求前缀和
    arg_sum_pallett = np.argwhere(sum_pallet >= maxDropNum)[0][0] # 前95%的颜色值数量;460

    maxNum = maxNum - arg_sum_pallett                             # 后5%的映射值数量
    num_values = num[1][::-1]                                     # 所有高频次颜色值的位置（颜色值降序）

    # 最终约简后的元素不能超过256个，不能少于10个
    maxNum = 256 if maxNum > 256 else maxNum
    if maxNum <= 10:
        maxNum = 10 if len(num) > 10 else len(num)

    # 将idx1i映射回RGB值
    color3i_init0 = (num_values / w[0]).astype(np.int32)
    color3i_init1 = (num_values % w[0]/w[1]).astype(np.int32)
    color3i_init2 = (num_values % w[1]).astype(np.int32)
    color3i = np.array([color3i_init0,color3i_init1,color3i_init2]).T  # (630, 3),按频率从小到大

    #=========================计算像素距离==============================
    zero2maxNum = color3i[:maxNum]                  #  5%的颜色值数量部分
    maxNum2len_Num = color3i[maxNum:]               # 95%的颜色值数量部分
    temp_matrix = np.zeros((len_num-maxNum,maxNum),dtype=np.int32)

    for i,single in enumerate(maxNum2len_Num):      # 分别求：95%的颜色值与5%的颜色值的距离
        temp_matrix[i] = np.sum(np.square(single-zero2maxNum), axis=1)

    arg_min = np.argmin(temp_matrix, axis=1)  # 找距离最小值像素代替原值
    replaceable_colors = num_values[arg_min]       # 通过索引获取5%的颜色值中距离95%的颜色值最近的颜色值

    pallet = dict(zip(num_values[:maxNum], range(maxNum)))
    # 遍历有效值
    for num_value,index_dist in zip(num_values[maxNum:],replaceable_colors):
        pallet[num_value] = pallet[index_dist]
    #=================================================================
    idx1i_reshape = idx1i.copy().reshape(1,-1)[0]
    idx1i_0 = np.zeros(height * width, dtype=np.int32)
    for i, v in enumerate(idx1i_reshape):
        idx1i_0[i] = pallet[v]
    idx1i = idx1i_0.reshape((height,width))
    color3f = np.zeros((1, maxNum, 3), np.float32)
    colorNum = np.zeros((1, maxNum), np.int32)
    np.add.at(color3f[0], idx1i, img3f)
    np.add.at(colorNum[0], idx1i, 1)
    colorNum_reshape = colorNum.T
    color3f[0] /= colorNum_reshape
    return color3f.shape[1],idx1i,color3f,colorNum