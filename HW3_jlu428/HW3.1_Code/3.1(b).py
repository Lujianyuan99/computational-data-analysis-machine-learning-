# @Time    : 2020/9/13 22:23
# @Author  : Jianyuan Lu
# @FileName: 3.1(b).py
# @Software: PyCharm


# @Time    : 2020/9/13 18:07
# @Author  : Jianyuan Lu
# @FileName: 3.1(a).py
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
data = data[:,[0,1]]
sampleNum = data.shape[0]
x1 = list(map(float, data[:,0]))
x2 = list(map(float, data[:,1]))
x1 = np.array(x1)
x2 = np.array(x2)
data = np.array([x1,x2]).T

min_data = data.min(0)
max_data = data.max(0)
nbin = 40        # you can change the number of bins in each dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=nbin)
temp = xedges[:-1]+xedges[1:]
xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
xpos = xpos.flatten()/2.
ypos = ypos.flatten()/2.
zpos = np.zeros_like (xpos)
dx = xedges [1] - xedges [0]
dy = yedges [1] - yedges [0]
myhist = hist/(dx*dy)
dz = myhist.flatten()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz )
plt.xlabel("amygdala")
plt.ylabel("acc")
plt.savefig("bar2D.jpg")



########################
# the following is KDE
########################


gridno = 40
inc1 = (max_data[0]-min_data[0])/gridno  #对应第一列,范围除以grid个数，小网格的长
inc2 = (max_data[1]-min_data[1])/gridno  #第二列，小网格的宽
gridx, gridy = np.meshgrid( np.arange(min_data[0], max_data[0]+0.5*inc1,inc1), np.arange(min_data[1], max_data[1]+0.5*inc2,inc2) )
gridall = [gridx.flatten(order = 'F'), gridy.flatten(order = 'F')] #flatten in column order,先把第一列排完，然后第二列
gridall = (np.asarray(gridall)).T
gridallno, nn= gridall.shape
norm_pdata = (np.power(data, 2)).sum(axis=1) #PCA后的结果，178个sample，2个维度
norm_gridall = (np.power(gridall, 2)).sum(axis=1)
cross = np.dot(data,gridall.T)  #交叉项
# compute squared distance between each data point and the grid point;
#dist2 = np.matlib.repmat(norm_pdata, 1, gridallno)
temp1 = np.repeat(norm_pdata, repeats =gridallno)
temp2 = np.tile(norm_gridall, sampleNum)
dist2 = np.repeat(norm_pdata, repeats =gridallno).reshape((len(norm_pdata), gridallno))\
        +np.tile(norm_gridall, sampleNum).reshape((len(norm_pdata), gridallno)) - 2* cross
#choose kernel bandwidth 1; please also experiment with other bandwidth;
#bandwidth = 1

#用likelihood求bandwidth
#bandwidth = cal_bandwidth(data)
bandwidth = 0.01
#evaluate the kernel function value for each training data point and grid
kernelvalue = np.exp(-dist2/(bandwidth*bandwidth*2))*(1/(2*math.pi))

#sum over the training data point to the density value on the grid points;
# here I dropped the normalization factor in front of the kernel function,
# and you can add it back. It is just a constant scaling;
temp = sum(kernelvalue)  #默认沿0轴方向
mkde = sum(kernelvalue) / (sampleNum*bandwidth*bandwidth)
#reshape back to grid;
mkde = ((mkde.T).reshape(gridno+1, gridno+1)).T



fig = plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot_surface(gridx, gridy, mkde)
ax.set_xlabel("amygdala")
ax.set_ylabel("acc")
plt.savefig("Guass2D.jpg")



