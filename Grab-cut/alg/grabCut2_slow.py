import sys
import os
import numpy as np
import math
import cv2 as cv
import igraph as ig
from sklearn.cluster import KMeans
import time
from sklearn import mixture
from maxflow import Graph
from .GMM import Model
# from Hist import Model

BG = 0  # 背景
FG = 1  # 前景
PR_BG = 2  # 未确定背景
PR_FG = 3  # 未确定前景
gamma = 30  # 平滑度项前面的系数

def score_formula(mult,mat):
    score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi)/np.sqrt(np.linalg.det(mat))
    return score

def bg_mask(mask):
    return np.where(np.logical_or(mask == BG, mask == PR_BG))  # 找mask中为0/2的坐标，即背景和未确定背景

def fg_mask(mask):
    return np.where(np.logical_or(mask == FG, mask == PR_FG))  # 找mask中为1/3的坐标，即背景和未确定背景

def construct_gc_graph(img,mask,gc_source,gc_sink,fgd_gmm,bgd_gmm,gamma,rows,cols,left_V,upleft_V,up_V,upright_V):
    # 由于拉平成一维了所以返回的tuple里只有一个ndarray，相当于序号而非坐标
    bgd_indexes = np.where(mask.reshape(-1) == BG)
    fgd_indexes = np.where(mask.reshape(-1) == FG)
    pr_indexes = np.where(np.logical_or(mask.reshape(-1) == PR_BG,mask.reshape(-1) == PR_FG))
    print('背景像素数: %d, 前景像素数: %d, 未确定像素数: %d' % (len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))

    node_count = cols * rows + 2
    gc_graph = Graph[float]()
    nodes = gc_graph.add_nodes(node_count)


    # 每个不确定像素和每个源点（对应背景）的权为-log(p(x))；和每个汇点（对应前景）权为-log(p(x))
    DS = -np.log(bgd_gmm.calc_prob(img.reshape(-1, 3)[pr_indexes]))
    DT = -np.log(fgd_gmm.calc_prob(img.reshape(-1, 3)[pr_indexes]))
    for node,ds,dt in zip(pr_indexes[0],DS,DT):
        gc_graph.add_tedge(node, ds, dt)

    # 每个背景像素有一条连边和每个源点（对应背景）的权为0，保证切割；和每个汇点（对应前景）权为1e8
    for node in bgd_indexes[0]:
        gc_graph.add_tedge(node, 0, 1e8)

    # 每个前景像素有一条连边和每个源点（对应背景）的权为0，保证切割；和每个汇点（对应前景）权为1e8
    for node in fgd_indexes[0]:
        gc_graph.add_tedge(node, 1e8, 0)

    # 右边的和左边的有一条连边
    img_indexes = np.arange(rows*cols,dtype=np.uint32).reshape(rows,cols)  # 制作每个像素的序号
    temp1 = img_indexes[:, 1:]
    temp2 = img_indexes[:, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1,n2,w in zip(mask1, mask2,left_V.reshape(-1).tolist()):
        gc_graph.add_edge(n1,n2,w,w)

    # 左上角和右下角有一条连边
    temp1 = img_indexes[1:, 1:]
    temp2 = img_indexes[:-1, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1,n2,w in zip(mask1, mask2,upleft_V.reshape(-1).tolist()):
        gc_graph.add_edge(n1,n2,w,w)

    # 下面的和上面的有一条连边
    temp1 = img_indexes[1:, :]
    temp2 = img_indexes[:-1, :]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1,n2,w in zip(mask1, mask2,up_V.reshape(-1).tolist()):
        gc_graph.add_edge(n1,n2,w,w)

    # 左下角和右上角有一条连边
    temp1 = img_indexes[1:, :-1]
    temp2 = img_indexes[:-1, 1:]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1,n2,w in zip(mask1, mask2,upright_V.reshape(-1).tolist()):
        gc_graph.add_edge(n1,n2,w,w)


    return gc_graph, nodes


def estimate_segmentation(mask,gc_graph,rows,cols,nodes):
    # 执行最小割算法
    T1 = time.time()
    flow = gc_graph.maxflow()
    T2 = time.time()
    print('最小割用时：',T2-T1)
    # 最小割结果后处理
    # mincut.partition分割的两部分的结果，[0]是前景，[1]是背景
    Iout = np.ones(shape=nodes.shape)
    for i in range(len(nodes)):
        Iout[i] = gc_graph.get_segment(nodes[i])
    bg_total=sum(Iout)
    fg_total=rows*cols-bg_total
    bg_indexs=np.where(Iout)
    print('最小割/能量函数取值：', flow)
    print('背景像素数: %d, 前景像素数: %d, ' % (bg_total, fg_total))

    pr_indexes = np.where(np.logical_or(mask == PR_BG, mask == PR_FG))  # 找掩码中为0/2的，即获取未确定像素的坐标
    img_indexes = np.arange(rows * cols,dtype=np.uint32).reshape(rows, cols)  # 制作每个像素的序号

    # 在原来不确定的像素中划分PR_FG(未确定前景)和PR_BG（未确定背景）
    mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], bg_indexs), PR_BG,PR_FG)
    bgd_indexes, fgd_indexes = bg_mask(mask), fg_mask(mask)  # 背景和前景的坐标

    print('可能的背景数为: %d, 可能的前景数为: %d' % (bgd_indexes[0].size, fgd_indexes[0].size))
    return pr_indexes, img_indexes, mask, bgd_indexes, fgd_indexes, flow


# 计算Beta以及普通边权
def cul_beta(img):
    rows, cols, _ = img.shape  # 图片行列数
    ################################################  求beta  ################################################
    # 求beta的准备工作
    _left_diff = img[:, 1:] - img[:, :-1]  # 右边像素减去左边像素的值
    _upleft_diff = img[1:, 1:] - img[:-1, :-1]  # 右下角像素减去左上角像素的值
    _up_diff = img[1:, :] - img[:-1, :]  # 下面的像素减去上面的像素的值
    _upright_diff = img[1:, :-1] - img[:-1, 1:]  # 左下角像素减去右上角像素的值
    # 对像素差开方
    sq_left_diff = np.square(_left_diff)  # shape:(342, 547, 3)=(H,W-1,3)
    sq_upleft_diff = np.square(_upleft_diff)  # shape:(341, 547, 3)=(H-1,W-1,3)
    sq_upright_diff = np.square(_upright_diff)  # shape:(341, 547, 3)=(H-1,W-1,3)
    sq_up_diff = np.square(_up_diff)  # shape:(341, 548, 3)=(H-1,W,3)

    # beta=1/(2*<(z_m-z_n)^2>) 经验值
    beta = np.sum(sq_left_diff) + np.sum(sq_upleft_diff) + np.sum(sq_up_diff) + np.sum(sq_upright_diff)
    beta = 1 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
    print('Beta:', beta)

    ################################################ 能量公式中的平滑度项V ################################################
    # 只是求出总体V，后面还要再根据alpha来选择哪些加入计算。注意这里沿着shpe:(342, 547)
    left_V = gamma * np.exp(-beta * np.sum(np.square(_left_diff), axis=2))
    upleft_V = gamma / np.sqrt(2) * np.exp(-beta * np.sum(np.square(_upleft_diff), axis=2))
    up_V = gamma * np.exp(-beta * np.sum(np.square(_up_diff), axis=2))
    upright_V = gamma / np.sqrt(2) * np.exp(-beta * np.sum(np.square(_upright_diff), axis=2))
    print(left_V.shape)

    return beta, left_V, upleft_V, up_V, upright_V


# 输入图像，掩码，矩形框；输出分割结果
def GrabCut(img, mask, rect):
    # 将图像转成矩阵
    img = np.asarray(img, dtype=np.float64)  # shape:(342, 548, 3)=(H,W,3)
    rows, cols, _ = img.shape  # 图片行列数
    if rect is not None:  # 矩形框非空，将掩码的矩形框范围内的值置为PR_FG（未确定前景）
        mask[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]] = PR_FG

    gc_source = cols*rows  # 源点序号（总数+1）
    gc_sink = gc_source + 1  # 汇点序号（总数+2）

    beta, left_V, upleft_V, up_V, upright_V=cul_beta(img)

    last = 1e9
    # 优化值小于上一次值的1%时停止迭代
    while(True):
        # 计算背景和前景的坐标，为一组tuple（里面包含两个ndarray，即对应像素的x,y坐标）
        bgd_indexes, fgd_indexes = bg_mask(mask), fg_mask(mask)
        print('可能的背景数量为: %d, 可能的前景数量为: %d' % (bgd_indexes[0].size, fgd_indexes[0].size))
        t5=time.time()
        bgd_gmm = GaussianMixture(img[bgd_indexes])  # 输入背景像素的序号，创建背景gmm
        fgd_gmm = GaussianMixture(img[fgd_indexes])  # 输入前景像素的序号，创建前景gmm
        t6=time.time()
        print('GMM初始化用时：', t6-t5)
        # bgd_gmm = mixture.GaussianMixture(n_components=5)  # 输入背景像素的序号，创建背景gmm
        # fgd_gmm = mixture.GaussianMixture(n_components=5)  # 输入前景像素的序号，创建前景gmm
        # bgd_gmm.fit(img[bgd_indexes])
        # fgd_gmm.fit(img[fgd_indexes])

        # 创建图
        T3 = time.time()
        gc_graph, nodes = construct_gc_graph(img,mask,gc_source,gc_sink,fgd_gmm,bgd_gmm,gamma,rows,cols,left_V,upleft_V,up_V,upright_V)
        T4 = time.time()
        print('建图用时为', T4-T3)
        # 分割
        pr_indexes,img_indexes,mask,bgd_indexes,fgd_indexes, value = estimate_segmentation(mask,gc_graph,rows,cols,nodes)
        if(last-value>0.01*last):
            last = value
        else:
            break
    return mask


