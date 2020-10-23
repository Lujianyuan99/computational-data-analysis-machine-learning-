# @Time    : 2020/8/23 19:42
# @Author  : Jianyuan Lu
# @FileName: Draw_Image.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from Tansfer_Picture_Data import *
from K_means import *
from K_metroid import *
import time
def draw_Image(points,m,original_data,vertical_digit,horizontal_digit):
    # 下面开始画图
    # plt.figure(figsize=(12, 9))
    # plt.imshow(original_data);  #和plt.show()配合才能画图
    # plt.axis('off')
    # plt.show()

    #建立输出文件夹
    if not np.os.path.exists("../Home1.2_Image"):
        np.os.mkdir("../Home1.2_Image")
    for i in [2,3,5,10,20,50]:
        # 建立输出文件夹
        filePackege = "../Home1.2_Image/K_equals" + str(i) + "/"
        if not np.os.path.exists(filePackege):
            np.os.mkdir(filePackege)
        print('Number of clusters:', i)
        starttime = time.time()
        #用K——means方法
        c_new, labels = K_means(points, i, m)
        # 用K——metroid方法
        #c_new, labels = K_metroid(points, i, m)
        endtime = time.time()
        runningTime = endtime - starttime
        runningtimeAnounce = "for K= " + str(i) + " the running time is " + str(runningTime)+"s"

        new_image = np.zeros((vertical_digit,horizontal_digit,3))
        labels = np.array(labels)
        j=0;
        #生成image三维矩阵
        while j<m:
            row = j // horizontal_digit;
            coloum = j % horizontal_digit
            new_image[row,coloum,:] = c_new[labels[j],:]
            j=j+1

        j = 0;
        #装换centroid,并画图
        new_centroid = np.zeros((1,i,3))
        while j < i:
            new_centroid[0,j,:]=c_new[j,:]
            j=j+1

        plt.figure(figsize=(12, 1))
        plt.imshow(new_centroid.astype('uint8'));
        plt.axis('off')
        fileaddressKdemo="../Home1.2_Image/"+str(i)+"K.jpg"
        plt.savefig(fileaddressKdemo)
        # plt.show()
        plt.figure(figsize=(12, 9))
        plt.imshow(new_image.astype('uint8'))
        plt.axis('off')
        plt.text(12,4,runningtimeAnounce)
        fileaddressdemo = filePackege + str(i) + "demo.jpg"
        plt.savefig(fileaddressdemo)
        # plt.show()