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
from getData import *



A = getData('../data/food-consumption.csv')



stdA = np.std(A,axis = 0)  #沿行计算标准差
stdA = skpp.normalize(stdA.reshape(1,-1)) # 使得所有标准差的平方和等于1  the normalize is different from MATLAB's
stdA2 = np.std(A,axis = 1)  #沿列计算标准差
stdA2= skpp.normalize(stdA2.reshape(1,-1))

meanA = (np.mean(A,axis = 0))
mean2A = (np.mean(A,axis = 1))
temp = A-meanA
temp2 = (A.T-mean2A)

Anew = (A-meanA) @ np.diag(np.ones(stdA.shape[1])/stdA[0])  #Anew每个元素除以对应的标准差
Anew2 = temp2 @ np.diag(np.ones(stdA2.shape[1])/stdA2[0])  #Anew每个元素除以对应的标准差

Anew = Anew.T  #归一化，转置后的A
Anew2 = Anew2.T
C = (Anew@Anew.T)/Anew.shape[1]
C2 = (Anew2@Anew2.T)/Anew2.shape[1]

K = 2
eigenvalue,eigenvector = ll.eigs(C,k = K)
eigenvalue = eigenvalue.real
eigenvector = eigenvector.real

eigenvalue2,eigenvector2 = ll.eigs(C2,k = K)
eigenvalue2 = eigenvalue2.real
eigenvector2 = eigenvector2.real

Z21 = (eigenvector2[:,0].T@Anew2)/math.sqrt(eigenvalue2[0])
Z22 =(eigenvector2[:,1].T@Anew2)/math.sqrt(eigenvalue2[1])



plt.stem(range(1,17),eigenvector2[:,0],linefmt = '-.', markerfmt = 'o', basefmt = '-')
plt.savefig("Z21.jpg")
plt.show()
plt.stem(range(1,17),eigenvector2[:,1],linefmt = '-.', markerfmt = 'o', basefmt = '-')
plt.savefig("Z22.jpg")
plt.show()
color_string = 'bgrmcky'
marker_string = '.+*ov^<>d'
leaf_fig2 = plt.figure()
i=0

while i < Anew2.shape[1]:
    print(i)
    color = color_string[i % 6]
    marker = marker_string[i % 9]
    m = color + marker
    leaf_fig2.gca().plot(Z21[i],Z22[i],m)   #m是图例，eg:m="g+"
    i = i+1
    plt.show()
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.savefig("scatter2.jpg")
plt.show()






temp3 = (eigenvector[:,0].T@Anew)
Z1 = (eigenvector[:,0].T@Anew)/math.sqrt(eigenvalue[0])
Z2 =(eigenvector[:,1].T@Anew)/math.sqrt(eigenvalue[1])
plt.stem(range(1,21),eigenvector[:,0],linefmt = '-.', markerfmt = 'o', basefmt = '-')
plt.savefig("Z1.jpg")

plt.stem(range(1,21),eigenvector[:,1],linefmt = '-.', markerfmt = 'o', basefmt = '-')
plt.savefig("Z2.jpg")


color_string = 'bgrmcky'
marker_string = '.+*ov^<>d'
leaf_fig = plt.figure()
i = 0
L = [];
while i < Anew.shape[1]:
    print(i)
    color = color_string[i % 6]
    marker = marker_string[i % 9]
    m = color + marker
    leaf_fig.gca().plot(Z1[i],Z2[i],m)   #m是图例，eg:m="g+"
    i = i+1
    plt.show()
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.savefig("scatter.jpg")
plt.show()