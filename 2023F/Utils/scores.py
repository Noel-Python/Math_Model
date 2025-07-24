import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import metrics

a = torch.randn((24, 24))
b = torch.randn((24, 24))

# 计算两图像间的MSE分数
def score_mse(x, y):
    score = np.mean((x - y)**2)
    return score

# 计算两图像间的SSIM分数
def score_ssim(x, y):
    score = metrics.structural_similarity(x, y, data_range=1.0)
    return score

# 计算两图像间的POD分数
def score_pod(x, y, threshold):
    x_bin = np.where(x >= threshold, 1, 0)
    y_bin = np.where(y >= threshold, 1, 0)
    na = np.sum(x_bin & y_bin)
    nb = np.sum((~x_bin) & y_bin)
    nc = np.sum(x_bin & (~y_bin))
    nd = np.sum((~x_bin) & (~y_bin))
    
    score = na / (na + nc)
    return score

# 计算两图像间的FAR分数
def score_far(x, y, threshold):
    x_bin = np.where(x >= threshold, 1, 0)
    y_bin = np.where(y >= threshold, 1, 0)
    na = np.sum(x_bin & y_bin)
    nb = np.sum((~x_bin) & y_bin)
    nc = np.sum(x_bin & (~y_bin))
    nd = np.sum((~x_bin) & (~y_bin))
    score = nb / (na + nb)
    return score

# 计算两图像间的CSI分数
def score_csi(x, y, threshold):
    x_bin = np.where(x >= threshold, 1, 0)
    y_bin = np.where(y >= threshold, 1, 0)
    na = np.sum(x_bin & y_bin)
    nb = np.sum((~x_bin) & y_bin)
    nc = np.sum(x_bin & (~y_bin))
    nd = np.sum((~x_bin) & (~y_bin))
    
    score = na / (na + nb + nc)
    return score

# 计算CAVR分数
def score_cavr(list_map: list, threshold):
    x = [list_map[0]]
    y = [list_map[-1]]
    sum = 0.0
    for i in range(len(list_map)):
        if i+5 < len(list_map):
            break
        c_t30 = np.sum(np.where(list_map[i+5] >= threshold, 1, 0))
        c_t0 = np.sum(np.where(list_map[i] >= threshold, 1, 0))
        sum += (c_t30 - c_t0)
    
    x_bin = np.where(x >= threshold, 1, 0)
    y_bin = np.where(y >= threshold, 1, 0)
    N = np.sum(x_bin & y_bin)
    
    score = (1000.0 * sum) / (256 * 256 * N)
    return score

if __name__ == "__main__":
    print(score_pod(a, b, threshold=0.5), score_far(a, b, threshold=0.35), score_csi(a, b, threshold=0.5))