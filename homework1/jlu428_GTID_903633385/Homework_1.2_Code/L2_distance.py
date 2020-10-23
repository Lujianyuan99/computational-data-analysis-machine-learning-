# @Time    : 2020/8/23 17:23
# @Author  : Jianyuan Lu
# @FileName: L2_distance.py
# @Software: PyCharm

#返回Label和centroid

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.sparse import csc_matrix, find
import random

def L2(c,points,m,K_number):
    #c元素squre，然后取负
    x2 = np.ones(3) * 2  #RBG有三个元素
    c2 = (np.sum(np.power(c, x2), axis=1,keepdims=True)).T
    points2 = (np.sum(np.power(points, x2), axis=1,keepdims=True))
    # For each data point x, computer min_j：  -2 * x' * c_j + c_j^2;简化版距离公式
    tempc1=np.power(c, np.ones(1))#转换数据类型
    #矩阵相乘，必须类型相同，浮点数对应浮点数，整数对应整数，不然会出错
    tmpdiff = -2 * np.dot(points, tempc1.T)+repmat(c2,m,1)+repmat(points2,1,K_number)

    # label_total = np.zeros((m,2))
    labels = (np.argmin(tmpdiff, axis=1).astype(int))
    # label_total[:,0] = labels
    # label_total[:,1] =(tmpdiff+repmat(c2,m,1)).min(1)
    # label_total = label_total.astype(int)

    #tmpdMaxValue=np.max(tmpdiff, axis=1)
    P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, K_number))
    #              稀疏矩阵的值，  row(i),col(i),把ones(m)放好，label是从0开始的
    count = P.sum(axis=0)


    c_new = np.array((P.T.dot(points)).T / count).T


    return c_new,labels