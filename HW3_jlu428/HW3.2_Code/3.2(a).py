# @Time    : 2020/9/16 16:39
# @Author  : Jianyuan Lu
# @FileName: 3.2(a).py
# @Software: PyCharm


from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans






a = loadmat('../data/data.mat')['data']
a = a.T
imageNumber = a.shape[0]
pixelEach = a.shape[1]

#生成数字的图片
# if not np.os.path.exists("../Home3.2_Number_Image"):
#     np.os.mkdir("../Home3.2_Number_Image")
#
# for i in range(0,imageNumber):
#     aReshape = np.reshape(a[:,i],(28,-1),order = 'F')
#     fig = plt.figure(figsize=(3, 3))
#     # Method1
#     ax1 = fig.add_subplot(111)
#     ax1.imshow(aReshape, cmap=plt.cm.gray)
#     fileaddressKdemo="../Home3.2_Number_Image/"+str(i)+".jpg"
#     plt.savefig(fileaddressKdemo)
#     plt.close('all')
#     print(i)a

aMean = np.sum(a,axis = 0)/imageNumber
ndata = a-aMean
C = np.matmul(ndata.T, ndata)/imageNumber

# pca the data
d = 5  # reduced dimension
V,aa,bb = np.linalg.svd(C) #奇异值分解
V = V[:, :d]
aa = aa[:d]
diagAA = np.diag(aa)

# project the data to the top 2 principal directions
pdata = np.dot(a,V)
###############
#6是从1032开始的
###############



# EM-GMM for wine data
# number of mixtures
K = 2
# random seed
seed = 5
pi = np.random.random(K)
pi = pi/np.sum(pi) #归一化，使得pi求和后等于1


mu = np.zeros(5)
mu.reshape(1,5)
sigmaMu = np.identity(5)
GuassMu = np.random.multivariate_normal(mu,sigmaMu,2)
GuassMu = GuassMu.T

Guasssigma = np.random.multivariate_normal(mu,sigmaMu,10)

sigma1 = Guasssigma[0:5,:]@Guasssigma[0:5,:].T+np.identity(5)
sigma2 = Guasssigma[5:10,:]@Guasssigma[5:10,:].T+np.identity(5)
#########################
#下面的sigma是final
sigma=[sigma1,sigma2]
mu = [GuassMu[:,0],GuassMu[:,1]]

tau = np.full((imageNumber, K), fill_value=0.)

# # parameter for countour plot
# xrange = np.arange(-5, -5, 0.1)
# yrange = np.arange(-5, -5, 0.1)

# ####
maxIter = 100
tol = 1e-3

mu = np.array(mu)
mu_old = mu.copy()
likelihood = np.zeros(100)


kmeans=KMeans(n_clusters=K)
kmeans.fit(pdata)
labelsKmeans = kmeans.labels_


misNumberKmeans = 0
for i in range(imageNumber):
    if i<1032 and labelsKmeans[i] ==0:
        misNumberKmeans = misNumberKmeans+1
    elif i>=1032 and labelsKmeans[i] ==1:
        misNumberKmeans = misNumberKmeans+1

for ii in range(100):

    # E-step
    for kk in range(K):
        tau[:, kk] = pi[kk] * mvn.pdf(pdata, mu[kk], sigma[kk])
    # normalize tau
    sum_tau = np.sum(tau, axis=1)
    sum_tau.shape = (imageNumber, 1)  # 有点像reshape
    tau = np.divide(tau, np.tile(sum_tau, (1, K)))

    # M-step
    for kk in range(K):
        # update prior
        pi[kk] = np.sum(tau[:, kk]) / imageNumber

        # update component mean
        mu[kk] = pdata.T @ tau[:, kk] / np.sum(tau[:, kk], axis=0)

        # update cov matrix
        dummy = pdata - np.tile(mu[kk], (imageNumber, 1))  # X-mu
        sigma[kk] = dummy.T @ np.diag(tau[:, kk]) @ dummy / np.sum(tau[:, kk], axis=0)

    temp = np.zeros((imageNumber, K))
    for jj in range(imageNumber):
        for kk in range(K):
            tempa = (pdata[jj, :] - mu[kk])
            tempb =(pdata[jj, :] - mu[kk]) @ np.linalg.inv(sigma[kk])@(pdata[jj, :] - mu[kk]).T
            temp[jj, kk] = (math.log(pi[kk]) - 0.5 * (pdata[jj, :] - mu[kk]) @ np.linalg.inv(sigma[kk]) \
                            @ (pdata[jj, :] - mu[kk]).T - 0.5 * math.log(np.linalg.det(sigma[kk])) \
                            - imageNumber / 2 * math.log(2 * math.pi)) * tau[jj, kk]

    temp1 = np.sum(temp,axis = 1)
    temp2 = np.sum(temp)
    likelihood[ii] = temp2

    print('-----iteration---', ii)

    if np.linalg.norm(mu - mu_old) < tol:
        likelihood = likelihood[0:ii+1]
        print('training coverged')
        break
    mu_old = mu.copy()
    if ii == 99:
        print('max iteration reached')
        break


mu = mu.T
aMean = np.tile(np.reshape(aMean,(aMean.shape[0],-1)),(1,K))
originalCenter = np.linalg.pinv(V.T)@(diagAA**(0.5)@mu)+aMean

labelsEM = np.zeros((tau.shape[0],1))
misNumber = 0
for i in range(tau.shape[0]):
    if tau[i,0] >= tau[i,1]:
        labelsEM[i] = 0
    else:
        labelsEM[i] = 1
    if i<1032 and labelsEM[i] ==0:
        misNumber = misNumber+1
    elif i>=1032 and labelsEM[i] ==1:
        misNumber = misNumber+1


for i in range(K):
    aReshape = np.reshape(originalCenter[:,i],(28,-1),order = 'F')
    fig = plt.figure(figsize=(3, 3))
    # Method1
    ax1 = fig.add_subplot(111)
    ax1.imshow(aReshape, cmap=plt.cm.gray)
    fileaddressKdemo=str(i)+".jpg"
    plt.savefig(fileaddressKdemo)
    plt.close('all')
    print(i)


for i in range(K):
    originalSigma = V@sigma[i]@V.T
    figure = plt.figure(figsize=(12, 10))
    axes = figure.add_subplot(111)
    # using the matshow() function
    caxes = axes.matshow(originalSigma, interpolation='nearest')
    figure.colorbar(caxes)
    figure.savefig("sigima"+str(i)+".jpg")
    figure.show()




# x2_smooth = np.linspace(0,likelihood.shape[0], 300)
# y2_smooth = make_interp_spline(np.array(range(likelihood.shape[0])), likelihood)(x2_smooth)
# #plt.bar(bin_edges2[:-1], myhist2,sbin2,align='center', alpha=0.5) #boundary+0.5 * sbin留出一定的空白
# plt.plot(x2_smooth, y2_smooth)
# plt.xlabel("iteration")
# plt.ylabel("log(likelihood)")
# plt.savefig("likelihood.jpg")
# plt.show()



