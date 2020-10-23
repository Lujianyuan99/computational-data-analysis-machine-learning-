# @Time    : 2020/9/13 20:45
# @Author  : Jianyuan Lu
# @FileName: KDE1D.py
# @Software: PyCharm

import math

def get_kde(x,data_array,bandwidth):

    def gauss(x):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*(x*x))

    N=len(data_array)
    res=0
    for i in range(N):  #求出某一个点的所有高斯叠加
        res += gauss((data_array[i]-x)/bandwidth)
    res /= (N*bandwidth)
    return res