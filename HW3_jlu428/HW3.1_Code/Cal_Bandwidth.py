# @Time    : 2020/9/14 11:08
# @Author  : Jianyuan Lu
# @FileName: Cal_Bandwidth.py
# @Software: PyCharm

import numpy as np
import math


def cal_bandwidth(data):
    def gauss(x): #传进来的是二维数组
        norm_x = (np.power(x, 2))
        temp = sum(norm_x)
        return (1/(2*math.pi))*math.exp(-0.5*sum(norm_x))


    h = np.arange(0.001,0.05,0.001)
    norm_h = (np.power(h, 2))
    norm_data = (np.power(data, 2))
    itera_Num = len(h)
    N=len(data)

    res= np.zeros_like (h)
    for i in range(itera_Num):  #求完所以h
        tempRes = np.zeros((N,1))
        for j in range(N):#求完所有点对特定的h
            for k in range(N):#求完这个点的
                tempRes[j] = tempRes[j] + (gauss((data[k, :] - data[j, :]) / h[i])) / (N * norm_h[i])
            tempRes[j] = math.log(tempRes[j])
        res[i] = sum(tempRes)


    return res