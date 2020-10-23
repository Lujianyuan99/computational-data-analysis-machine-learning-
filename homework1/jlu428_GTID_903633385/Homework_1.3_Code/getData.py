# @Time    : 2020/8/25 21:19
# @Author  : Jianyuan Lu
# @FileName: getData.py
# @Software: PyCharm

import os
import numpy as np
from os.path import abspath, exists
from scipy import sparse
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def getData(fileaddress):
    # read the graph from 'play_graph.txt'
    f_path = abspath(fileaddress)
    if exists(f_path):
        with open(f_path) as graph_file:  #可能不存在这个file path,异常，中断。with as可以避免
            lines = [line.split() for line in graph_file]  #空格，split
    return np.array(lines).astype(int)