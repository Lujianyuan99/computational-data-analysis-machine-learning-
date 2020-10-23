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

path = '../data/n90pol.csv'

data = pd.read_csv(path,header=None).to_numpy()
data = data[1:,:]
sampleNum = data.shape[0]
x1 = list(map(float, data[:,0]))
x2 = list(map(float, data[:,1]))
x1 = np.array(x1)
x2 = np.array(x2)
nbin = 30

min1 = min(x1)
max1 = max(x1)
sbin1 = (max1 - min1)/ nbin


min2 = min(x2)
max2 = max(x2)
sbin2 = (max2 - min2)/ nbin #每个bin的宽度


hist1,bin_edges1 = numpy.histogram(x1, bins=nbin)
hist2,bin_edges2 = numpy.histogram(x2, bins=nbin)

print(sum(hist1))
myhist1 = hist1*(1/sbin1)/90
myhist2 = hist2*(1/sbin2)/90



temp = bin_edges1[:-1]
plt.figure()
plt.bar(bin_edges1[:-1], myhist1,sbin1,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
plt.xlabel("amygdala")
plt.savefig("BarForAmygdala.jpg")
plt.show()


# ##基本可以替代上面那一串。。。。
# plt.figure()
# plt.hist(x1,nbin,density=True)
# plt.savefig("BarForAcc.jpg")
# plt.show()



plt.figure()
plt.bar(bin_edges2[:-1], myhist2,sbin2,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
plt.xlabel("acc")
plt.savefig("BarForAcc.jpg")
plt.show()


########################
# the following is KDE
########################


bandwidth1 = 1.06*np.std(x1)*pow(sampleNum,-0.2)
bandwidth2 = 1.06*np.std(x2)*pow(sampleNum,-0.2)

x1plot = np.array(list(set(x1.tolist())))
index = x1plot.argsort()
x1plot = x1plot[index]

x2plot = np.array(list(set(x2.tolist())))
index = x2plot.argsort()
x2plot = x2plot[index]

y1 = np.zeros_like(x1plot)
y2 = np.zeros_like(x2plot)

for i in range(len(x1plot)):
    y1[i] = get_kde(x1plot[i],x1,bandwidth1)

for i in range(len(x2plot)):
    y2[i] = get_kde(x2plot[i],x2,bandwidth2)


x1_smooth = np.linspace(x1plot.min(), x1plot.max(), 300)
y1_smooth = make_interp_spline(x1plot, y1)(x1_smooth)
plt.figure()
#plt.bar(bin_edges1[:-1], myhist1,sbin1,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
plt.plot(x1_smooth, y1_smooth)
plt.xlabel("amygdala")
plt.savefig("GuassAmm.jpg")
plt.show()


x2_smooth = np.linspace(x2plot.min(), x2plot.max(), 300)
y2_smooth = make_interp_spline(x2plot, y2)(x2_smooth)
#plt.bar(bin_edges2[:-1], myhist2,sbin2,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
plt.plot(x2_smooth, y2_smooth)
plt.xlabel("acc")
plt.savefig("GaussAcc.jpg")
plt.show()




