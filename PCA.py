#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Time:2018/9/21
PCA 实现
"""
import numpy as np
import matplotlib.pyplot as plt
def build_data(n=200):
    """
    构建10维数据，并且期望将其降到三维。
    :return:
    """
    data = []
    main=[1,3,5]
    for i in range(n):
        index = np.random.randint(0, 3)
        k =main[index]
        noise = np.random.randn(1,1000)
        array1 = [k+part for part in noise]
        data.append(list(array1[0]))
    return data

def pca(data,n_components):
    meandata = np.mean(data,axis=0)
    meanremoved = data-meandata
    convdata = np.cov(meanremoved,rowvar=False)
    enval,envec = np.linalg.eig(np.mat(convdata))
    ensortind = np.argsort(enval)
    enval = ensortind[:-(n_components+1):-1]
    nedenvec = envec[:,enval]
    lowdata = meanremoved*nedenvec
    newdata = (lowdata*nedenvec.T)+meandata
    return lowdata,newdata

data = build_data()
lowdata,newdata = pca(data,2)
data = np.asarray(data)
newdata = np.asarray(newdata)
lowdata = np.asarray(lowdata)
plt.scatter(data[:,0],data[:,1],marker='H',c='red')
plt.scatter(newdata[:,0],newdata[:,1],marker='1',c='blue')

plt.show()

