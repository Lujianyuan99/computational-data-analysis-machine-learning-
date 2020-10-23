# @Time    : 2020/10/5 15:17
# @Author  : Jianyuan Lu
# @FileName: Part2b.py
# @Software: PyCharm

# @Time    : 2020/10/4 22:16
# @Author  : Jianyuan Lu
# @FileName: Part1b.py
# @Software: PyCharm


import csv
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


a = loadmat('data.mat')['data']
a = a.T
imageNumber = a.shape[0]
pixelEach = a.shape[1]
data = a
label = np.arange(0, imageNumber, 1)
for i in range(0,imageNumber):
    if i<1032:
        label[i] = 2
    else:
        label[i] = 6

pca = PCA(n_components=2)
#用X来训练模型pca，训练好后(建立好了映射)，transform用来降维
Xpca = pca.fit(data).transform(data) #fit就是学习的过程，PCA是无监督学习，所以没有y.fit函数返回调用fit函数的本身



h = .02  # step size in the mesh

names = ["K Nearest Neighbors",
         "Naive Bayes", "LogisticRegression"]

classifiers = [
    KNeighborsClassifier(3),
    GaussianNB(),
    LogisticRegression()]

figure = plt.figure(figsize=(27, 9))

x_min, x_max = Xpca[:, 0].min() - .5, Xpca[:, 0].max() + .5
y_min, y_max = Xpca[:, 1].min() - .5, Xpca[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

i = 1
test_size0=0.5
Xtrain, Xtest, ytrain, ytest = train_test_split(Xpca, label, test_size=test_size0, random_state=1)
#下面是话input data
cm = plt.cm.RdBu  # denotes a colormap
cm_bright = ListedColormap(['#FF0000', '#0000FF']) #自定义颜色
ax = plt.subplot(1, len(classifiers) + 1, i)  # subplot(numRows, numCols, plotNum)

ax.set_title("Input data")
legend = ["trainLabel2","trainLabel6","testLabel2","testLabel6"]


numTrain2 = np.sum(ytrain ==2)
numTrain6 = np.sum(ytrain ==6)
numTest2 = np.sum(ytest ==2)
numTest6 = np.sum(ytest ==6)
Xtrain2 = np.zeros((numTrain2,2))
Xtrain6 = np.zeros((numTrain6,2))
Xtest2 = np.zeros((numTest2,2))
Xtest6 = np.zeros((numTest6,2))
ytrain2 = np.zeros((numTrain2,1))
ytrain6 = np.zeros((numTrain6,1))
ytest2 = np.zeros((numTest2,1))
ytest6 = np.zeros((numTest6,1))

k2=0
k6=0
for j in range(len(ytrain)):
    if ytrain[j] == 2:
        Xtrain2[k2] = Xtrain[j]
        ytrain2[k2] = ytrain[j]
        k2 = k2+1
    else:
        Xtrain6[k6] = Xtrain[j]
        ytrain6[k6] = ytrain[j]
        k6=k6+1
k2=0
k6=0
for j in range(len(ytest)):
    if ytest[j] == 2:
        Xtest2[k2] = Xtest[j]
        ytest2[k2] = ytest[j]
        k2 = k2+1
    else:
        Xtest6[k6] = Xtest[j]
        ytest6[k6] = ytest[j]
        k6 = k6+1

ax.scatter(Xtrain2[:, 0], Xtrain2[:, 1], c='#FF0000', cmap=cm_bright,\
                edgecolors='k',label = legend[0])
ax.scatter(Xtrain6[:, 0], Xtrain6[:, 1], c='#0000FF', cmap=cm_bright,\
                edgecolors='k',label = legend[1])
ax.scatter(Xtest2[:, 0], Xtest2[:, 1], c='#FF0000', cmap=cm_bright,alpha=0.6,\
                edgecolors='k',label = legend[2])
ax.scatter(Xtest6[:, 0], Xtest6[:, 1], c='#0000FF', cmap=cm_bright,alpha=0.6,\
                edgecolors='k',label = legend[3])
ax.legend(loc='upper right', shadow=False, scatterpoints=1)


ax.set_xlim(xx.min(), xx.max())  # 设置x轴的数值显示范围。
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1



# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(Xtrain, ytrain)
    if hasattr(clf, "sigma_"):
        k=0
        j=0
        while(k<clf.sigma_.shape[0]):
            while(j<clf.sigma_.shape[1]):
                if clf.sigma_[k][j]<0.001:
                    clf.sigma_[k][j] = 0.001
                j = j+1
            k = k+1
    score = clf.score(Xtest, ytest)#拟合优度越大，说明x对y的解释程度越高。自变量对因变量的解释程度越高，自变量引起的变动占总变动的百分比高。观察点在回归直线附近越密集。

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"): #判断有没有这个function，有的clif没有
               #判断每个点的颜色显示
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  #ravel函数，遍历矩阵各个元素

    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8) #画三维等高线，contourf()会填充轮廓

    # Plot the training points
    # ax.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, cmap=cm_bright,
    #            edgecolors='k')
    # # Plot the testing points
    # ax.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest, cmap=cm_bright,
    #            edgecolors='k', alpha=0.6)
    ax.scatter(Xtrain2[:, 0], Xtrain2[:, 1], c='#FF0000', cmap=cm_bright, \
               edgecolors='k', label=legend[0])
    ax.scatter(Xtrain6[:, 0], Xtrain6[:, 1], c='#0000FF', cmap=cm_bright, \
               edgecolors='k', label=legend[1])
    ax.scatter(Xtest2[:, 0], Xtest2[:, 1], c='#FF0000', cmap=cm_bright, alpha=0.6, \
               edgecolors='k', label=legend[2])
    ax.scatter(Xtest6[:, 0], Xtest6[:, 1], c='#0000FF', cmap=cm_bright, alpha=0.6, \
               edgecolors='k', label=legend[3])
    ax.legend(loc='upper right', shadow=False, scatterpoints=1)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.savefig("scatterPart2_"+str(test_size0)+".jpg")
plt.show()

