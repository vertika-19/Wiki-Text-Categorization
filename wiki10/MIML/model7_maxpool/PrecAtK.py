from DataParser import DataParser as DataParser
from model7_maxpool import Model7_maxpool as Model
import numpy as np
import os, sys
import math

def ComputePrecisionK(modelfile,testfile,outputfile):
    maxParagraphLength = int(sys.argv[1])
    maxParagraphs = int(sys.argv[2] )
    filterSizes = [int(i) for i in sys.argv[3].split("-")]
    num_filters = int(sys.argv[4])
    wordEmbeddingDimension = int(sys.argv[5])
    lrate = float(sys.argv[10])

    keep_prob = 1.0

    labels = 30938
    vocabularySize = 101939

    model = Model(maxParagraphs,maxParagraphLength,labels,vocabularySize,\
        filterSizes,num_filters,wordEmbeddingDimension,lrate, keep_prob)

    testing = DataParser(maxParagraphs,maxParagraphLength,labels,vocabularySize)
    testing.getDataFromfile(testfile)

    model.load(modelfile)

    print("loading done")
    print("no of test examples: " + str(testing.totalPages))

    print("Computing Prec@k")
    
    #check if batchsize needs to be taken by parameter

    batchSize = 1
    testing.restore()
    truePre=[]
    pred=[]
    for itr in range(testing.totalPages):
        data=testing.nextBatch(1)
        truePre.append(data[0])
        pre=model.predict(data)
        pred.append(pre[0])

    K_list = [1,3,5]     #prec@1 .....prec@NoofLabels
    precAtK = [0.0]*6	

    # #As need to get Prec only on last 50% of test data as first 50% is for cross validation
    # valid=int(len(truePre)*0.5)
    # pred = pred[valid:]
    # truePre = truePre[valid:]

    for i,v in enumerate(pred):
        temp = [(labId,labProb) for labId,labProb in enumerate(v) ]
        temp = sorted(temp,key=lambda x:x[1],reverse=True)  #sorting based on label probability to get top k
        for ele in K_list:        #1....No of Labels
            pBag = 0              #no of true positive for this instance 
            for itr in range(ele): #top k ie top ele
                if truePre[i][0][temp[itr][0]]==1:
                	precAtK[ele] += 1 
                    # pBag += 1
            # precAtK[ele] += float(pBag)/float(ele)

    f = open(outputfile,"a")
    output = sys.argv[9]

    for k in K_list:
		precAtK[k] /= (k * len(pred)) 
		print ("Prec@" + str(k) + " = " + str(precAtK[k]))
		output = output + "," + "Prec@" + str(k) + "=," + str(precAtK[k])
    f.write(output + "\n")
    f.close()

if __name__ == '__main__':
    testfile = "../wiki10_miml_test.txt"

    modelfile = "models/" + sys.argv[8] + "/model7_maxpool_wiki10_" + sys.argv[7]
    outputfile = "results/precAtk.txt"
    ComputePrecisionK(modelfile,testfile,outputfile)
