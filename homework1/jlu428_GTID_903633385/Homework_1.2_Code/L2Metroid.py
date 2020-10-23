# @Time    : 2020/8/25 11:29
# @Author  : Jianyuan Lu
# @FileName: L2Metroid.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.sparse import csc_matrix, find
import sys
import random

def L2_metroid(c,points,m,K_number):
    #c元素squre，然后取负
    x2 = np.ones(3) * 2  #RBG有三个元素
    c2 = (np.sum(np.power(c, x2), axis=1,keepdims=True)).T
    points2 = (np.sum(np.power(points, x2), axis=1,keepdims=True))

    tempc1=np.power(c, np.ones(1))#转换数据类型

    tmpdiff = -2 * np.dot(points, tempc1.T)+repmat(c2,m,1)+repmat(points2,1,K_number)

    label_total = np.zeros((m,2))
    labels = (np.argmin(tmpdiff, axis=1).astype(int))
    label_total[:,0] = labels
    label_total[:,1] =tmpdiff.min(1)
    label_total = label_total.astype(np.int32)

    clusterRBG = np.full((K_number,3), sys.maxsize)
    clusterMinValue = np.full((K_number,1), sys.maxsize)

    i =0
    while i<m:    #i是点的标号
        if label_total[i,1]<clusterMinValue[label_total[i,0],0]:
            clusterRBG[label_total[i,0],:] = points[i,:]
            clusterMinValue[label_total[i, 0], 0] = label_total[i,1]
        i = i+1

    return clusterRBG,labels