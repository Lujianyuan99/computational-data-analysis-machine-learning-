# @Time    : 2020/8/25 10:13
# @Author  : Jianyuan Lu
# @FileName: L1_distancs.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.sparse import csc_matrix, find
import random

def L1(c,points,m,K_number):
    points = points.astype(np.int32)
    c = c.astype(np.int32)
    reshapeC = np.reshape(c, (1, K_number,3))
    reshapePoints = np.reshape(points, (m,1,3))
    tmpdiff = np.sum(np.abs(np.repeat(reshapePoints, K_number, axis=1)-np.repeat(reshapeC, m, axis=0)),axis=2)

    labels = (np.argmin(tmpdiff, axis=1).astype(int))

    P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, K_number))
    #              稀疏矩阵的值，  row(i),col(i),把ones(m)放好，label是从0开始的
    count = P.sum(axis=0)

    c_new = np.array((P.T.dot(points)).T / count).T


    return c_new,labels