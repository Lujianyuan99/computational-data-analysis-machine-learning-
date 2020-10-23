# @Time    : 2020/8/23 16:51
# @Author  : Jianyuan Lu
# @FileName: Tansfer_Picture_Data.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

def transfer_Data(file_address):
    original_data = plt.imread(file_address)
    vertical =original_data.shape[0]
    horizontal =original_data.shape[1]
    RBG_number =original_data.shape[2]
    point = np.reshape(original_data, (vertical * horizontal, RBG_number))
    #number of point
    m = point.shape[0]
    return m, point, original_data,vertical,horizontal