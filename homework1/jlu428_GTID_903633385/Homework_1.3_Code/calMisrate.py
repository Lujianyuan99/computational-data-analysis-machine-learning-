# @Time    : 2020/8/26 10:30
# @Author  : Jianyuan Lu
# @FileName: calMisrate.py
# @Software: PyCharm


import os
import numpy as np
from os.path import abspath, exists
from scipy import sparse
from scipy.sparse import find
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from getData import *
from filterIsolatedDate import *
from getWebName import *
from calMisrate import *
from findMajority import *


def calMisrate(c_idx,orientation,n):

    AVGmisrate = np.zeros((1,c_idx.shape[1])).astype(float)
    i = 0
    while i<c_idx.shape[1]:
        clusterBelong = np.zeros((c_idx.shape[0], i+2)).astype(int)
        majorityEach = np.zeros((2, i+2)).astype(int)
        minorityNumber = np.zeros((1, i+2)).astype(int)

        j = 0
        while j < c_idx.shape[0]:
            clusterBelong[j,c_idx[j,i]] = 1
            majorityEach[orientation[j],c_idx[j,i]] = majorityEach[orientation[j],c_idx[j,i]]+1
            j = j+1
        majority = np.argmax(majorityEach, axis=0).astype(int)

        minorityNumber = np.min(majorityEach,axis=0)
        rateEach = np.true_divide(minorityNumber, np.sum(clusterBelong,axis=0))
        AVGmisrate[0,i] = (rateEach@(np.sum(clusterBelong,axis=0)).T)/n
        i = i+1


    return AVGmisrate
