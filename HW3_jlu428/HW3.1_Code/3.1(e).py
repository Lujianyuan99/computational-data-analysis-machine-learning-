# @Time    : 2020/9/15 14:39
# @Author  : Jianyuan Lu
# @FileName: 3.1(e).py
# @Software: PyCharm

import csv
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from KDE1D import *
from scipy.interpolate import make_interp_spline
from Cal_Bandwidth import *
from sklearn.neighbors.kde import KernelDensity




path = '../data/n90pol.csv'

data = pd.read_csv(path,header=None).to_numpy()
data = data[1:,:]
x1 = list(map(float, data[:,0]))
x2 = list(map(float, data[:,1]))
x3 = list(map(float, data[:,2]))
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
data = np.array([x1,x2,x3]).T

indexList = []
for c in range(2,6):
    indexList.append([])

for c in range(2,6):
    for i in range(data.shape[0]):
        if data[i,2] == c:
            indexList[c-2].append(i)

for c in range(2,6):
    index = np.array(indexList[c-2])
    dataI = data[index,:]
    xNew = dataI[:,[0,1]]

    min_data = xNew.min(0)
    max_data = xNew.max(0)
    sampleNum = xNew.shape[0]

    gridno = 40
    inc1 = (max_data[0] - min_data[0]) / gridno  # 对应第一列,范围除以grid个数，小网格的长
    inc2 = (max_data[1] - min_data[1]) / gridno  # 第二列，小网格的宽
    gridx, gridy = np.meshgrid(np.arange(min_data[0], max_data[0] + 0.5 * inc1, inc1),
                               np.arange(min_data[1], max_data[1] + 0.5 * inc2, inc2))
    gridall = [gridx.flatten(order='F'), gridy.flatten(order='F')]  # flatten in column order,先把第一列排完，然后第二列
    gridall = (np.asarray(gridall)).T
    gridallno, nn = gridall.shape
    norm_pdata = (np.power(xNew, 2)).sum(axis=1)  # PCA后的结果，178个sample，2个维度
    norm_gridall = (np.power(gridall, 2)).sum(axis=1)
    cross = np.dot(xNew, gridall.T)  # 交叉项
    # compute squared distance between each data point and the grid point;
    # dist2 = np.matlib.repmat(norm_pdata, 1, gridallno)
    temp = np.repeat(norm_pdata, repeats=gridallno)
    print(len(norm_pdata))
    print(gridallno)
    temp = temp.reshape((len(norm_pdata), gridallno))
    temp2 = np.tile(norm_gridall, sampleNum).reshape((len(norm_pdata), gridallno))
    dist2 = np.repeat(norm_pdata, repeats=gridallno).reshape((len(norm_pdata), gridallno)) \
            + np.tile(norm_gridall, sampleNum).reshape((len(norm_pdata), gridallno)) - 2 * cross
    # choose kernel bandwidth 1; please also experiment with other bandwidth;
    # bandwidth = 1

    # 用likelihood求bandwidth
    # bandwidth = cal_bandwidth(data)
    bandwidth = 0.01
    # evaluate the kernel function value for each training data point and grid
    kernelvalue = np.exp(-dist2 / (bandwidth * bandwidth * 2)) * (1 / (2 * math.pi))

    # sum over the training data point to the density value on the grid points;
    # here I dropped the normalization factor in front of the kernel function,
    # and you can add it back. It is just a constant scaling;
    temp = sum(kernelvalue)  # 默认沿0轴方向
    mkde = sum(kernelvalue) / (sampleNum * bandwidth * bandwidth)
    # reshape back to grid;
    mkde = ((mkde.T).reshape(gridno + 1, gridno + 1)).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("amygdala")
    ax.set_ylabel("acc")
    ax.plot_surface(gridx, gridy, mkde)
    plt.savefig("./1eFigure/joint"+str(c)+".jpg")
    plt.show()