# @Time    : 2020/10/4 12:02
# @Author  : Jianyuan Lu
# @FileName: Part2a.py
# @Software: PyCharm


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


path = 'marriage.csv'

data = pd.read_csv(path,header=None).to_numpy()
label = data[:,-1]
data = data[:,0:-1]
Xtrain, Xtest, ytrain, ytest = train_test_split(data, label, test_size=0.2, random_state=0)

h = .02  # step size in the mesh

names = ["K Nearest Neighbors",
         "Naive Bayes", "LogisticRegression"]

classifiers = [
    KNeighborsClassifier(3),
    GaussianNB(),
    LogisticRegression()]

for name,clType in zip(names, classifiers):
    clf = clType.fit(Xtrain, ytrain)
    if hasattr(clf, "sigma_"):
        i=0
        j=0
        while(i<clf.sigma_.shape[0]):
            while(j<clf.sigma_.shape[1]):
                if clf.sigma_[i][j]<0.001:
                    clf.sigma_[i][j] = 0.001
                j = j+1
            i = i+1
    ypred_train = clf.predict(Xtrain)
    matched_train = ypred_train == ytrain
    acc_train = sum(matched_train)/len(matched_train)


    # training confusion matrix
    idx1 = np.where(ytrain ==0) #原数据0的位置
    idx2 = np.where(ytrain ==1)

    cf_train_00 = np.sum(ytrain[idx1] == ypred_train[idx1])
    cf_train_01 = np.sum(ypred_train[idx1] ==1)
    cf_train_11 = np.sum(ytrain[idx2] == ypred_train[idx2])
    cf_train_10 = np.sum(ypred_train[idx2] ==0)

    # ## test error
    ypred_test = clf.predict(Xtest)
    matched_test = ypred_test == ytest
    acc_test = sum(matched_test)/len(matched_test)

    # testing confusion matrix
    idx1 = np.where(ytest ==0)
    idx2 = np.where(ytest ==1)
    cf_test_00 = np.sum(ytest[idx1] == ypred_test[idx1])
    cf_test_01 = np.sum(ypred_test[idx1] ==1)
    cf_test_11 = np.sum(ytest[idx2] == ypred_test[idx2])
    cf_test_10 = np.sum(ypred_test[idx2] ==0)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("the classifier name: " + name)
    print('the training accuracy: '+ str(round(acc_train, 4)))
    print('confusion matrix for training:')
    print('          predected 1       predected 2')
    print(f"true 0        {cf_train_00}                {cf_train_01}")
    print(f"true 1        {cf_train_10}                {cf_train_11}")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("the classifier name: " + name)
    print('the testing accuracy: '+ str(round(acc_test, 4)))
    print('confusion matrix for testing:')
    print('          predected 1       predected 2')
    print(f"true 0        {cf_test_00}                {cf_test_01}")
    print(f"true 1        {cf_test_10}                {cf_test_11}")