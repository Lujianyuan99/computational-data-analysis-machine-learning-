# @Time    : 2020/8/23 17:08
# @Author  : Jianyuan Lu
# @FileName: K_means.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from L2_distance import *
from L1_distancs import *

def K_means(points,K_number,m):
    # Randomly initialize centroids with data points;
    # 生成随机的centroid位置，二维数组
    # size是随机数输出尺寸，一行二列
    uniques,index = np.unique(points.astype(int),return_index = True,axis = 0)
    temp = np.random.permutation(index)[:K_number]
    c = points[temp,:]
    #用L1距离
    #c_new, labels = L1(c, points, m, K_number)
    #用L2距离
    c_new, labels=L2(c,points,m,K_number)
    iteration = 0
    while np.sum((np.sum(np.power(c-c_new, 2), axis=0)))>10 or iteration<200:
        c = c_new
        #用L1距离
        #c_new, labels = L1(c, points, m, K_number)
        # 用L2距离
        c_new, labels=L2(c,points,m,K_number)
        iteration = iteration+1
        #print(iteration)


    return c_new,labels