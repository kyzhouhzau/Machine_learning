#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
from  sklearn import metrics
class BiClassification(object):
    """
    计算二分类时的P值，R值，F值。
    以及加权调和F值

    """
    @staticmethod
    def PRFScore(truelabel,predictlabel,labeltype):
        assert isinstance(truelabel,list)
        assert isinstance(predictlabel,list)
        TP=0
        FP=0
        FN=0
        TN=0
        small=1e-15
        labelkinds = labeltype
        for i,label in enumerate(truelabel):
            if predictlabel[i] in labelkinds:
                if label == labelkinds[0] and label == predictlabel[i]:
                    TP+=1
                elif label == labelkinds[1] and label== predictlabel[i]:
                    TN+=1
                elif label == labelkinds[0] and label != predictlabel[i]:
                    FP+=1
                elif label == labelkinds[1] and label != predictlabel[i]:
                    FN+=1


        P = TP/(TP+FP+small)
        R = TP/(TP+FN+small)
        F = 2*P*R/(P+R+small)
        return P,R,F

    @staticmethod
    def Fbata(truelabel, predictlabel,labeltype,bata=1):
        """
        :param truelabel:
        :param predictlabel:
        :param bata:0-1倾向查准率 ，1-inf倾向查全率
        :return:
        """
        assert isinstance(truelabel, list)
        assert isinstance(predictlabel, list)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        small = 1e-10
        labelkinds = labeltype
        for i, label in enumerate(truelabel):

            if label == labelkinds[0] and label == predictlabel[i]:
                TP += 1
            elif label == labelkinds[1] and label == predictlabel[i]:
                TN += 1
            elif label == labelkinds[0] and label != predictlabel[i]:
                FP += 1
            elif label == labelkinds[1] and label != predictlabel[i]:
                FN += 1
        P = TP / (TP + FP + small)
        R = TP / (TP + FN + small)
        F = (1+bata**2) * P * R / (bata**2*P + R)
        return P, R, F

def main():
    truelabel = [1,2,1,1,1,2,2,1,1,1,2,2,1,2,2,1,1,1]
    predictlabel = [1,2,1,2,1,2,1,1,1,1,2,2,1,2,1,2,1,2]

    P,R,F = BiClassification.PRFScore(truelabel,predictlabel,[1,2])
    _,_,bata = BiClassification.Fbata(truelabel,predictlabel,[1,2],3)

    z = metrics.f1_score(truelabel,predictlabel,average="macro")
    x= metrics.recall_score(truelabel,predictlabel,average="macro")
    matrix = metrics.confusion_matrix(truelabel,predictlabel)
    report = metrics.classification_report(truelabel,predictlabel)
    # print(x)
    #
    # print(F)
    # print(z)
    # print(bata)
    print(matrix)
    print(report)
if __name__=="__main__":
    main()
