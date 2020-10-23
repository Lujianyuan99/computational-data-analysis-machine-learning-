# @Time    : 2020/9/6 20:37
# @Author  : Jianyuan Lu
# @FileName: homework2.1.py
# @Software: PyCharm

from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
from L2distance import *
from Matrix_D import *
from L1distance import *

a = loadmat('../data/isomap.mat')['images']
imageNumber = a.shape[1]
pixelEach = a.shape[0]

#生成雕像的图片
# if not np.os.path.exists("../Home2.2_Sculpture_Image"):
#     np.os.mkdir("../Home2.2_Sculpture_Image")
#
# for i in range(369,imageNumber):
#     aReshape = np.reshape(a[:,i],(64,-1),order = 'F')
#     fig = plt.figure(figsize=(3, 3))
#     # Method1
#     ax1 = fig.add_subplot(111)
#     ax1.imshow(aReshape, cmap=plt.cm.gray)
#     fileaddressKdemo="../Home2.2_Sculpture_Image/"+str(i)+".jpg"
#     plt.savefig(fileaddressKdemo)
#     print(i)

#L2距离
A = AfromL2(a,imageNumber)
#L1距离
#A = AfromL1(a,imageNumber)

A[A==float(0)] = float('inf')
I = np.diag(np.ones(imageNumber))
oneVector = np.ones((imageNumber,1))

temp = oneVector@(oneVector.T)/imageNumber
H = I-oneVector@(oneVector.T)/imageNumber
D = Matrix_D(A)
C = H@(D**2)@H/(-2)

K =2
eigenvalue,eigenvector = ll.eigs(C,k = K)

#np.savetxt('A_test.csv', A, delimiter = ',')
eigenvalue = eigenvalue.real
eigenvector = eigenvector.real
eigenvalueDiag = np.diag(eigenvalue**0.5)
Z = eigenvector@eigenvalueDiag

fig,ax=plt.subplots()
x = Z[:,0]
y = Z[:,1]
ax.scatter(x,y,c='r')
for i in range(0,imageNumber,10):
    ax.annotate(i,(x[i],y[i]))

plt.savefig("scatter_10L1distance.jpg")




