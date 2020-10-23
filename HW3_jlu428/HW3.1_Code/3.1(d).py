# @Time    : 2020/9/15 13:50
# @Author  : Jianyuan Lu
# @FileName: 3.1(d).py
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
sampleNum = data.shape[0]
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
    x11 = dataI[:,0]
    x22 = dataI[:, 1]
    bandwidth1 = 1.06 * np.std(x11) * pow(dataI.shape[0], -0.2)
    bandwidth2 = 1.06 * np.std(x22) * pow(dataI.shape[0], -0.2)

    x1plot = np.array(list(set(x11.tolist())))
    index1 = x1plot.argsort()
    x1plot = x1plot[index1]

    x2plot = np.array(list(set(x22.tolist())))
    index2 = x2plot.argsort()
    x2plot = x2plot[index2]

    y1 = np.zeros_like(x1plot)
    y2 = np.zeros_like(x2plot)

    for i in range(len(x1plot)):
        y1[i] = get_kde(x1plot[i], x11, bandwidth1)

    for i in range(len(x2plot)):
        y2[i] = get_kde(x2plot[i], x22, bandwidth2)

    x1_smooth = np.linspace(x1plot.min(), x1plot.max(), 300)
    y1_smooth = make_interp_spline(x1plot, y1)(x1_smooth)
    plt.figure()
    # plt.bar(bin_edges1[:-1], myhist1,sbin1,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
    plt.plot(x1_smooth, y1_smooth)
    plt.xlabel("amygdala")
    plt.savefig("./1dFigure/Amm"+str(c)+".jpg")
    plt.show()

    x2_smooth = np.linspace(x2plot.min(), x2plot.max(), 300)
    y2_smooth = make_interp_spline(x2plot, y2)(x2_smooth)
    # plt.bar(bin_edges2[:-1], myhist2,sbin2,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
    plt.plot(x2_smooth, y2_smooth)
    plt.xlabel("acc")
    plt.savefig("./1dFigure/Acc"+str(c)+".jpg")
    plt.show()