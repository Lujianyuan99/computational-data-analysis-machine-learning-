# @Time    : 2020/8/26 10:04
# @Author  : Jianyuan Lu
# @FileName: getWebName.py
# @Software: PyCharm

import os
import numpy as np
from os.path import abspath, exists

def getwebname(fileaddress):
    # read inverse_teams.txt file
    f_path = abspath(fileaddress)
    idx2name = []
    idxOrientation = []
    if exists(f_path):
        with open(f_path) as fid:
            for line in fid.readlines():
                #str.split( );       # 以空格为分隔符，有多少分隔多少
                #str.split(' ', 1 ); # 以空格为分隔符，分隔成两个
                name = line.split("\t")[3]
                orientation = line.split("\t")[2]
                idx2name.append(name[:-1])
                idxOrientation.append(orientation)
    return np.array(idx2name),np.array(idxOrientation).astype(int)