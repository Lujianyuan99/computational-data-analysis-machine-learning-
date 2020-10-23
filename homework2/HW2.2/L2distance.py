# @Time    : 2020/9/7 21:48
# @Author  : Jianyuan Lu
# @FileName: L2distance.py
# @Software: PyCharm


import numpy as np

import matplotlib.pyplot as plt


def AfromL2(a,imageNumber):
    A = np.zeros((imageNumber, imageNumber))
    epsilong = 502
    k = 0
    while k < float('inf'):
    # while k < 1:
        i = 0
        while i < imageNumber:
            j = i + 1
            while j < imageNumber:
                distanceij = (a[:, i] - a[:, j]).T @ (a[:, i] - a[:, j])
                #print(distanceij)
                if distanceij <= epsilong:
                    A[i, j] = distanceij
                j = j + 1
            i = i + 1
        A = A + A.T
        template = np.ones((imageNumber, 1))
        positionMatrix = A != float(0)
        z = positionMatrix @ template
        epsilong = epsilong + 10
        k = k + 1
        print(k)
        print(z.min())
        if z.min() >= 100:
            break

    A = A**0.5


    figure = plt.figure(figsize=(12, 10))
    axes = figure.add_subplot(111)

    # using the matshow() function
    caxes = axes.matshow(A, interpolation='nearest')
    figure.colorbar(caxes)
    figure.savefig("L2A.jpg")


    return A