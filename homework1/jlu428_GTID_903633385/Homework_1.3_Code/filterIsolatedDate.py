# @Time    : 2020/8/25 21:47
# @Author  : Jianyuan Lu
# @FileName: filterIsolatedDate.py
# @Software: PyCharm

import os
import numpy as np
from os.path import abspath, exists
from scipy import sparse
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def filterData(A,webName,orientation):

    sum = np.sum(A, axis=1)
    zeros = np.where(sum == 0)
    temp = np.array(zeros[0])
    A = np.delete(A, temp, axis=0)
    A = np.delete(A, temp, axis=1)
    webName = np.delete(webName, temp)
    orientation =np.delete(orientation, temp)
    return A,zeros,webName,orientation