# @Time    : 2020/8/25 21:19
# @Author  : Jianyuan Lu
# @FileName: homework1.3.py
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




n=1490
largest_K =101

edgeData = getData('../data/edges.txt')
p1 = edgeData[:, 0] - 1  # 矩阵是从0开始的
p2 = edgeData[:, 1] - 1

v = np.ones((edgeData.shape[0], 1)).flatten()  # 生成一堆1，准备存放

#A = sparse.csc_matrix((v, (p1, p2)), shape=(n, n))
temp = np.zeros((n, n)).astype(int)
temp[p1,p2] = 1
A = temp
A = (A + np.transpose(A)) / 2  # 连接的地方都是0.5
webName,orientation = getwebname('../data/nodes.txt')
A, zeroIndex,webName,orientation = filterData(A,webName,orientation)
n=A.shape[0]
temp = np.sum(A, axis=1)
D = np.diag(1 / np.sqrt(np.sum(A, axis=1)))  # 不用这个.A1，就生成不了矩阵
L = D @ A @ D  # 执行矩阵乘法

v, x = np.linalg.eig(L)  # v是特征值，x特征向量
v1 = v.real  # 手动安排降序排列\
x = x.real
small_V_index = np.argsort(-v1) #然后直接去x里面找
c_idx = np.zeros((A.shape[0], largest_K-2)).astype(int)


for k in range(2,largest_K):
    x_temp = x[:, small_V_index[0:k]]  # 每列对应是特征向量
    # 逐元素点乘     -1是自动计算行，1是列
    x_temp = x_temp / np.repeat(np.sqrt(np.sum(x_temp * x_temp, axis=1).reshape(-1, 1)), k, axis=1)
    kmeans = KMeans(n_clusters=k).fit(x_temp)
    c_idx[:,k-2] = kmeans.labels_

AVGmsirate = calMisrate(c_idx,orientation,n)


plt.plot(range(2,largest_K), AVGmsirate[0,:], color="r", linestyle="-", linewidth=1)
plt.title("The mismatch rate variation with K increase")
plt.xlabel("Clustering K number")
plt.ylabel("AVG Mismatch Rate")
plt.savefig("Mismatch.jpg")
plt.show()




