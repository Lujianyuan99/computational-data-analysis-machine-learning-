# @Time    : 2020/9/8 20:40
# @Author  : Jianyuan Lu
# @FileName: normalPCA.py
# @Software: PyCharm

# @Time    : 2020/9/3 11:23
# @Author  : Jianyuan Lu
# @FileName: Homework2.1.py
# @Software: PyCharm


import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
from scipy.io import loadmat

A = loadmat('isomap.mat')['images']
imageNumber = A.shape[1]


Anew2 = A

C2 = (Anew2@Anew2.T)/Anew2.shape[1]  #求协方差矩阵

K = 2

eigenvalue2,eigenvector2 = ll.eigs(C2,k = K)
eigenvalue2 = eigenvalue2.real
eigenvector2 = eigenvector2.real

Z21 = (eigenvector2[:,0].T@Anew2)/math.sqrt(eigenvalue2[0])
Z22 =(eigenvector2[:,1].T@Anew2)/math.sqrt(eigenvalue2[1])



fig,ax=plt.subplots()
x = Z21
y = Z22
ax.scatter(x,y,c='r')
for i in [320, 650, 640,600,410,170,210,520,660,400,240,270,510,80,\
          290,620,670,420]:
    ax.scatter(x[i], y[i], color='b', edgecolors='g')

# for i in range(0,imageNumber,10):
#     ax.annotate(i,(x[i],y[i]))

plt.savefig("PCANoMark.jpg")
