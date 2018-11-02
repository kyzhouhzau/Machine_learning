#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Boosting Tree实现
@Author:zhoukaiyin
"""
import numpy as np
def get_ms(offset,data_y):
    c1s = [];c2s=[]
    for i,_ in enumerate(data_x):
        if i+1<offset:
            c1s.append(data_y[i])
        else:
            c2s.append(data_y[i])
    c1 = sum(c1s)/(len(c1s)+1e-10)
    c2 = sum(c2s)/len(c2s)
    ms = sum([(c-c1)**2 for c in c1s])+sum([(c-c2)**2 for c in c2s])
    return ms,round(c1,2),round(c2,2)

class Boost_Tree():
    def __init__(self,train_x,train_y,margin=0.2):
        self.cut_margin= []
        self.tsita = []
        self.train_x = train_x
        self.train_y = train_y
        while True:
            L = self.boost_tree()
            if L <margin:
                break

    def mse(self,residuals):
        L = sum([x**2 for x in residuals])
        return L

    def boost_tree(self):
        cut_points = [(self.train_x[i]+self.train_x[i+1])/2 for i in range(len(data_x)-1)]
        mss = []
        for offset in cut_points:
            ms,_,_ = get_ms(offset, self.train_y)
            mss.append(ms)
        offset= cut_points[mss.index(min(mss))]
        self.cut_margin.append(offset)
        _,c1,c2 = get_ms(offset,self.train_y)
        self.tsita.append((c1,c2))
        residuals = []
        for x in self.train_x:
            index = x-1
            if x<offset:
                residual = round(self.train_y[index]-c1,2)
                residuals.append(residual)
            else:
                residual = round(self.train_y[index]-c2,2)
                residuals.append(residual)
        self.train_y=residuals
        L = self.mse(residuals)
        return L

def build_T(testx,offsetlist,tsitalist):
    FX = [[m,tsitalist[i][0],tsitalist[i][1]] for i,m in enumerate(offsetlist)]
    base_tree = lambda x, v, f1, f2: x < v and f1 or f2
    reg_efunc = lambda x: sum([base_tree(x, v, f1, f2) for v, f1, f2 in FX])
    reg_func = np.frompyfunc(reg_efunc, 1, 1)
    regress_result = reg_func(testx)
    return regress_result

if __name__=="__main__":
    """
    这个案例是李航统计学习方法第149页例题。
    """
    data_x = [1,2,3,4,5,6,7,8,9,10]
    data_y = [5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9.0,9.05]
    bt = Boost_Tree(data_x,data_y)
    print(bt.cut_margin)
    print(bt.tsita)
    regress_result = build_T([5,1,3,4,5], bt.cut_margin, bt.tsita)
    print(regress_result)


