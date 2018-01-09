#https://stackoverflow.com/questions/9004172/precision-recall-for-multiclass-multilabel-classification
from DataParser import DataParser as DataParser
from model3 import Model3 as Model
import numpy as np
import os, sys
import math


def ComputePrecisionK(modelfile,testfile,K_list):

    maxParagraphLength=10
    maxParagraphs=4
    #nlabels=1001
    #vocabularySize=76391
    labels=8
    vocabularySize=244
    model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)

    testing = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
    print(testfile)
    testing.getDataFromfile(testfile)
    print("data loading done")
    print("no of test examples: " + str(testing.totalPages))

    model.load(modelfile)

    print("model loading done")

    batchSize = 1

    testing.restore()
    truePre=[]
    pred=[]
    for itr in range(testing.totalPages):
        data=testing.nextBatch(1)
        truePre.append(data[0])
        pre=model.predict(data)
        pred.append(pre[0])

    precAtK={}
    for itr in K_list:
        precAtK[itr]=0

    for i,v in enumerate(pred):
        temp=[(labId,labProb) for labId,labProb in enumerate(v) ]
    #     print(temp)
        temp=sorted(temp,key=lambda x:x[1],reverse=True)
        for ele in K_list:
            pBag=0
            for itr in range(ele):
                if truePre[i][0][temp[itr][0]]==1:
                    pBag+=1
        #         print(float(pBag)/float(ele))
            precAtK[ele]+=float(pBag)/float(ele)

    f=open("results/precAtK_model3_n","w")
    for key in sorted(precAtK.keys()):
    #     print(key, precAtK[key]/len(pred))
        print(precAtK[key]/len(pred))
        f.write(str(key)+"\t"+ str(precAtK[key]/len(pred))+"\n")
    f.close()

if __name__ == '__main__':
    modelfile = "/home/khushboo/Desktop/miml/models/model3_reuter_90"
    testfile = "/home/khushboo/Desktop/Reuter_dataset/reuters_sparse_test.txt"
    K_list = [1,2,3,4,5,6,7]

    ComputePrecisionK(modelfile,testfile,K_list)
