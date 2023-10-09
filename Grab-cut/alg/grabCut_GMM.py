import numpy as np
from sklearnex import patch_sklearn
patch_sklearn() #启动加速补丁

from sklearn.cluster import KMeans
import time
from sklearn.datasets import make_blobs
from maxflow import Graph
from .GMM import Model
# from Hist import Model

BG = 0  # 背景
FG = 1  # 前景
PR_BG = 2  # 未确定背景
PR_FG = 3  # 未确定前景
gamma = 30  # 平滑度项前面的系数

def bg_mask(mask):
    return np.where(np.logical_or(mask == BG, mask == PR_BG))  # 找mask中为0/2的坐标，即背景和未确定背景

def fg_mask(mask):
    return np.where(np.logical_or(mask == FG, mask == PR_FG))  # 找mask中为1/3的坐标，即背景和未确定背景


def construct_gc_graph(gc_graph,img,mask,fgd_model,bgd_model):
    # 由于拉平成一维了所以返回的tuple里只有一个ndarray，相当于序号而非坐标
    bgd_indexes = np.where(mask.reshape(-1) == BG)
    fgd_indexes = np.where(mask.reshape(-1) == FG)
    pr_indexes = np.where(np.logical_or(mask.reshape(-1) == PR_BG,mask.reshape(-1) == PR_FG))
    print('背景像素数: %d, 前景像素数: %d, 未确定像素数: %d' % (len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))

    # 每个不确定像素和每个源点（对应前景）的权为-log(p(x))；和每个汇点（对应背景）权为-log(p(x))
    DS = -np.log(bgd_model.calc_prob(img.reshape(-1, 3)[pr_indexes])+1e-8)
    DT = -np.log(fgd_model.calc_prob(img.reshape(-1, 3)[pr_indexes])+1e-8)

    for node,ds,dt in zip(pr_indexes[0],DS,DT):
        gc_graph.add_tedge(node, ds, dt)
    return gc_graph


def construct_base_graph(mask, left_V, upleft_V, up_V, upright_V):
    rows, cols = mask.shape
    node_count = cols * rows
    base_graph = Graph[float]()
    nodes = base_graph.add_nodes(node_count)

    bgd_indexes = np.where(mask.reshape(-1) == BG)
    fgd_indexes = np.where(mask.reshape(-1) == FG)
    # 每个背景像素有一条连边和每个源点的权为0，保证切割；和每个汇点权为1e8
    for node in bgd_indexes[0]:
        base_graph.add_tedge(node, 0, 1e8)

    # 每个前景像素有一条连边和每个汇点的权为0，保证切割；和每个源点权为1e8
    for node in fgd_indexes[0]:
        base_graph.add_tedge(node, 1e8, 0)

    # 右边的和左边的有一条连边
    img_indexes = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)  # 制作每个像素的序号
    temp1 = img_indexes[:, 1:]
    temp2 = img_indexes[:, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1, n2, w in zip(mask1, mask2, left_V.reshape(-1).tolist()):
        base_graph.add_edge(n1, n2, w, w)

    # 左上角和右下角有一条连边
    temp1 = img_indexes[1:, 1:]
    temp2 = img_indexes[:-1, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1, n2, w in zip(mask1, mask2, upleft_V.reshape(-1).tolist()):
        base_graph.add_edge(n1, n2, w, w)

    # 下面的和上面的有一条连边
    temp1 = img_indexes[1:, :]
    temp2 = img_indexes[:-1, :]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1, n2, w in zip(mask1, mask2, up_V.reshape(-1).tolist()):
        base_graph.add_edge(n1, n2, w, w)

    # 左下角和右上角有一条连边
    temp1 = img_indexes[1:, :-1]
    temp2 = img_indexes[:-1, 1:]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    for n1, n2, w in zip(mask1, mask2, upright_V.reshape(-1).tolist()):
        base_graph.add_edge(n1, n2, w, w)

    return base_graph,nodes

def estimate_segmentation(mask,gc_graph,rows,cols,nodes):
    # 执行最小割算法
    flow = gc_graph.maxflow()

    Iout = np.ones(shape=nodes.shape)
    for i in range(len(nodes)):
        Iout[i] = gc_graph.get_segment(nodes[i])
    bg_indexs = np.where(Iout)  # 返回True代表汇点，即背景点（print输出发现前面的都是True，而前面的都是矩形框外的背景）
    print('最小割的值：', flow)
    pr_indexes = np.where(np.logical_or(mask == PR_BG, mask == PR_FG))  # 找掩码中为0/2的，即获取未确定像素的坐标
    img_indexes = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)  # 制作每个像素的序号

    # 在原来不确定的像素中划分PR_FG(未确定前景)和PR_BG（未确定背景）
    mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], bg_indexs), PR_BG,PR_FG)
    bgd_indexes, fgd_indexes = bg_mask(mask), fg_mask(mask)  # 背景和前景的坐标

    print('可能的背景数为: %d, 可能的前景数为: %d' % (bgd_indexes[0].size, fgd_indexes[0].size))
    return mask, flow


# 计算Beta以及普通边权
def pre_cul(img):
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

    return left_V, upleft_V, up_V, upright_V


# 输入图像，掩码，矩形框；输出分割结果
def GrabCut(img, mask, rect):
    # 将图像转成矩阵
    img = np.asarray(img, dtype=np.float32)  # shape:(342, 548, 3)=(H,W,3)
    rows, cols, _ = img.shape  # 图片行列数
    if rect is not None:  # 矩形框非空，将掩码的矩形框范围内的值置为PR_FG（未确定前景）
        mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = PR_FG
    # 提前计算V项
    left_V, upleft_V, up_V, upright_V = pre_cul(img)

    # 创建基本无向图
    base_graph, nodes = construct_base_graph(mask, left_V, upleft_V, up_V, upright_V)
    last, i = 1e9, 0
    while(True):
        i += 1
        bgd_indexes, fgd_indexes = bg_mask(mask), fg_mask(mask)
        # 创建前景，背景model
        bgd_model, fgd_model = Model(img[bgd_indexes]), Model(img[fgd_indexes])
        # 创建无向图
        gc_graph = construct_gc_graph(base_graph.copy(), img, mask, fgd_model, bgd_model)
        # 分割
        mask, value = estimate_segmentation(mask, gc_graph, rows, cols, nodes)
        if(last-value<0.01*last or i>=10):
            break
        else:
            last = value
    return mask


