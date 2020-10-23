# @Time    : 2020/8/22 23:03
# @Author  : Jianyuan Lu
# @FileName: Homework1.2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from Tansfer_Picture_Data import *
from K_means import *
from Draw_Image import *




m, points,original_data,vertical_digit,horizontal_digit = transfer_Data('../data/football.bmp')


draw_Image(points,m,original_data,vertical_digit,horizontal_digit)






