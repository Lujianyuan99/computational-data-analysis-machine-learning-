# @Time    : 2020/10/21 19:48
# @Author  : Jianyuan Lu
# @FileName: 2a.py
# @Software: PyCharm

import csv
from random import sample

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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,recall_score,f1_score,precision_score
from sklearn.svm import SVC
from sklearn import metrics
import math



a = loadmat('mnist_10digits.mat')
xtrain = a["xtrain"]
ytrain = a["ytrain"]
xtest = a["xtest"]
ytest = a["ytest"]

# imageNumber = xtrain.shape[0]
# if not np.os.path.exists("../Home_Number_Image"):
#     np.os.mkdir("../Home_Number_Image")

# for i in range(0,imageNumber):
#     aReshape = np.reshape(xtrain[i,:],(28,-1),order = 'F')
#     fig = plt.figure(figsize=(3, 3))
#     # Method1
#     ax1 = fig.add_subplot(111)
#     ax1.imshow(aReshape, cmap=plt.cm.gray)
#     fileaddressKdemo="../Home_Number_Image/"+str(i)+".jpg"
#     plt.savefig(fileaddressKdemo)
#     plt.close('all')
#     print(i)
#     if i>50 :
#         break

xtrain = xtrain/255

ytrain = ytrain.reshape(ytrain.shape[1:])
ytest = ytest.reshape(ytest.shape[1:])


sample1 = sample(range(0,60000),5000)
xtrainSample = xtrain[sample1,:]
ytrainSample = ytrain[sample1]


sample2 = sample(range(0,60000),1000)
xtrainSample2 = xtrain[sample2,:]
ytrainSample2 = ytrain[sample2]




# following is to calculate sigma
i = 0
j = 0
k = 0
myarray = np.zeros(999000)
while(i<1000):
    while(j<1000):
        if(i !=j):
            myarray[k]=np.sum(np.square(xtrainSample2[i,:]-xtrainSample2[j,:]))
            k = k+1
        j= j+1
    i = i+1
    j=0
sigma = math.sqrt(np.median(myarray)/2)

classifiers = [
    # KNeighborsClassifier(10),
    # LogisticRegression(),
    # SVC(kernel='linear'),
    SVC(gamma=0.0001),
    # MLPClassifier(hidden_layer_sizes=(20,10),max_iter=500,activation='relu',
    #               alpha=0.0001, batch_size='auto', beta_1=0.9,
    #            beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #             learning_rate='constant',learning_rate_init=0.001, momentum=0.9,
    #            nesterovs_momentum=True, power_t=0.5, random_state=None,
    #            shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
    #            verbose=False, warm_start=False)]
    ]
for classifier in classifiers:
    trained_Model = classifier.fit(xtrainSample,ytrainSample)
    expected = ytest
    predicted = classifier.predict(xtest)

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

