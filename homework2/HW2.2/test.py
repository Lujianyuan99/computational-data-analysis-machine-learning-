# @Time    : 2020/9/7 11:54
# @Author  : Jianyuan Lu
# @FileName: test.py
# @Software: PyCharm

import numpy as np
from numpy import random as rd
from pylab import *
from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
from L2distance import *
from Matrix_D import *
import matplotlib.pyplot as plt

a = loadmat('../data/isomap.mat')['images']
a100 = a[:,100]
print(a100)

aReshape = np.reshape(a100,(64,-1))






print(True*1)

x = rd.randint(-5,5,(5,5))
y = x==3 #得到boolean矩阵

z = x[y]
print (z.size)
print (x[x==3])
b=0
