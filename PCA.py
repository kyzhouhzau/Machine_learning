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
    return data#生成(1,1000)的数据

def pca(data,n_components):
    meandata = np.mean(data,axis=0)#竖着方向
    meanremoved = data-meandata
    convdata = np.cov(meanremoved,rowvar=False)#计算协方差矩阵
    enval,envec = np.linalg.eig(np.mat(convdata))#np.mat()将序列转为np的二维数组，np.linalg.eig返回特征值，和特征向量。
    ensortind = np.argsort(enval)#排序并返回排序后的下标。
    enval = ensortind[:-(n_components+1):-1]#反向选取index即选取最大的几个index
    nedenvec = envec[:,enval]#取这几个特征值对应的特征向量。
    lowdata = meanremoved*nedenvec#
    newdata = (lowdata*nedenvec.T)+meandata
    return newdata

data = build_data()
newdata = pca(data,2)
data = np.asarray(data)
newdata = np.asarray(newdata)
plt.scatter(data[:,0],data[:,1],marker='H',c='red')
plt.scatter(newdata[:,0],newdata[:,1],marker='1',c='blue')

plt.show()

