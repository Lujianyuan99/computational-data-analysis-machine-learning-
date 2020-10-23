# @Time    : 2020/8/25 11:29
# @Author  : Jianyuan Lu
# @FileName: L1Metroid.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.sparse import csc_matrix, find
import random
import sys

def L1metroid(c,points,m,K_number):
    points = points.astype(np.int32)
    c = c.astype(np.int32)
    reshapeC = np.reshape(c, (1, K_number,3))
    reshapePoints = np.reshape(points, (m,1,3))
    tmpdiff = np.sum(np.abs(np.repeat(reshapePoints, K_number, axis=1)-np.repeat(reshapeC, m, axis=0)),axis=2)

    label_total = np.zeros((m, 2))
    labels = (np.argmin(tmpdiff, axis=1).astype(int))
    label_total[:, 0] = labels
    label_total[:, 1] = tmpdiff.min(1)
    label_total = label_total.astype(np.int32)

    clusterRBG = np.full((K_number, 3), sys.maxsize)
    clusterMinValue = np.full((K_number, 1), sys.maxsize)
    i =0
    while i<m:    #i是点的标号
        if label_total[i,1]<clusterMinValue[label_total[i,0],0]:
            clusterRBG[label_total[i,0],:] = points[i,:]
            clusterMinValue[label_total[i, 0], 0] = label_total[i,1]
        i = i+1

    return clusterRBG,labels