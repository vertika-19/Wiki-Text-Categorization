from DataParser import DataParser as DataParser
from model7 import Model7 as Model
import numpy as np
from sklearn.metrics import confusion_matrix
import os, sys
import math

def genAnalysis(modelfile,testfile,outputfile):
    maxParagraphLength = 20
    maxParagraphs = 5
    filterSizes = [1]
    num_filters = 64
    wordEmbeddingDimension = 30
    lrate = float(1e-3)
    labels = 30938
    vocabularySize = 101939

    model = Model(maxParagraphs,maxParagraphLength,labels,vocabularySize,filterSizes,num_filters,wordEmbeddingDimension,lrate)

    testing = DataParser(maxParagraphs,maxParagraphLength,labels,vocabularySize)
    testing.getDataFromfile(testfile)

    model.load(modelfile)

    print("loading done")
    print("no of test examples: " + str(testing.totalPages))

    batchSize = 1
    testing.restore()
    truePre=[]
    pred=[]
    for itr in range(testing.totalPages):
        data=testing.nextBatch(1)
        truePre.append(data[0])
        pre=model.predict(data)
        pred.append(pre[0])

    labelIDName = open("../labelId-labelName-full.txt").read().split("\n")
    labelIDName = [  [ int(x.split("\t")[0]) , x.split("\t")[1].rstrip() ] for x in labelIDName]
    # print(labelIDName)    

    #making it a dictionary
    labelName = dict(labelIDName)
    # print(labelName[9026])

    f = open(outputfile,"w")
    for i,v in enumerate(pred):
        temp = [(labId,labProb) for labId,labProb in enumerate(v) ]
        temp = sorted(temp,key=lambda x:x[1],reverse=True)  #sorting based on label probability to get top k
        predLabel = [0]*len(temp)

        output = ""
        for itr in range(11):
            predLabel[temp[itr][0]] = 1
            if truePre[i][0][temp[itr][0]] == 1:
                output = output + "," + labelName[temp[itr][0]]
        f.write(str(i) + ","  + output + "\n")
    f.close()

if __name__ == '__main__':
    testfile = "../wiki10_miml_test.txt"
    modelfile = "/home/khushboo/Desktop/MTP/temp_model7/M7_20_5_1_64_30_1_400_0.01/model7_wiki10_100"
    outputfile = "results/top10labels_analysis.txt"
    genAnalysis(modelfile,testfile,outputfile)
