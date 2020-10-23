# @Time    : 2020/8/26 11:16
# @Author  : Jianyuan Lu
# @FileName: findMajority.py
# @Software: PyCharm

import numpy as np

def findMajority(clusterBelong,orientation):
    majorityEach = np.zeros((1, clusterBelong.shape[1])).astype(int)
    i0 = 0
    i1 = 0
    i = 0