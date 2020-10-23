# @Time    : 2020/9/8 19:01
# @Author  : Jianyuan Lu
# @FileName: L1distance.py
# @Software: PyCharm

# @Time    : 2020/9/7 21:48
# @Author  : Jianyuan Lu
# @FileName: L2distance.py
# @Software: PyCharm


import numpy as np

import matplotlib.pyplot as plt


def AfromL1(a,imageNumber):
    A = np.zeros((imageNumber, imageNumber))
    epsilong = 1010
    k = 0
    while k < float('inf'):
    #while k < 1:
        i = 0
        while i < imageNumber:
            j = i + 1
            while j < imageNumber:
                distanceij = np.linalg.norm(a[:, i] - a[:, j], ord=1, axis=None, keepdims=False)
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
    figure.savefig("L1A.jpg")


    return A