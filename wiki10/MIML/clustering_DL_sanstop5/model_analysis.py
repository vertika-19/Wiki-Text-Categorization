from DataParser import DataParser as DataParser
from clustering_DL import clustering_DL as Model
import numpy as np
import os, sys
import math
def genAnalysis(modelfile,testfile,outputfile):
    maxParagraphLength = 50
    maxParagraphs = 10
    filterSizes = [2,3,4]
    num_filters = 64
    wordEmbeddingDimension = 50
    lrate = float(0.001)
    poolLength = 5
    labels = 30938
    vocabularySize = 101939

    keep_prob = 1.0

    model = Model(maxParagraphs,maxParagraphLength,labels,vocabularySize,\
                    filterSizes,num_filters,wordEmbeddingDimension,lrate,poolLength, keep_prob)

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

    labelIDName = open("labelId-labelName-full.txt").read().strip().split("\n")
    labelIDName = [  [ int(x.strip().split("\t")[0]) , x.strip().split("\t")[1].rstrip() ] for x in labelIDName]
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
        for itr in range(5):
			output = output + "," + labelName[temp[itr][0]]
        
            #predLabel[temp[itr][0]] = 1
            #if truePre[i][0][temp[itr][0]] == 1:
            #    output = output + "," + labelName[temp[itr][0]]
        f.write(str(i) + ","  + output + "\n")
    f.close()

if __name__ == '__main__':
    testfile = "../../dataset/original_split/wiki10_miml_test.txt"
    modelfile = "models/M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5/clustering_DL_300"
    outputfile = "results/top10labels_analysis.txt"
    genAnalysis(modelfile,testfile,outputfile)
