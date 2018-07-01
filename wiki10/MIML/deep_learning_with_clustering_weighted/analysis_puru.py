from DataParser import DataParser as DataParser
from model7 import Model7 as Model
import numpy as np
import os, sys
import math
import matplotlib.pyplot as plt

#M7_20_10_2-3-4_64_100_100_100_0.001_2     

def analyse(modelfile,testfile,outputfile):
    maxParagraphLength = 20
    maxParagraphs = 10
    filterSizes = [2,3,4]
    num_filters = 64
    wordEmbeddingDimension = 100
    lrate = float(0.001)
    poolLength = 2
    labels = 30938
    vocabularySize = 101939

    model = Model(maxParagraphs,maxParagraphLength,labels,vocabularySize,\
                    filterSizes,num_filters,wordEmbeddingDimension,lrate,poolLength)

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

    labelids = open("../../dataset/sorted_labelid_sans5toplabels.txt","r").read().strip().split("\n")
    labelids = [ int(x) for x in labelids ]

    no_of_partition = 10
    partition_size = labels / no_of_partition
    rank1 = [0]*no_of_partition
    rank3 = [0]*no_of_partition
    rank5 = [0]*no_of_partition

    for i,v in enumerate(pred):
        temp = [(labId,labProb) for labId,labProb in enumerate(v) ]
        temp = sorted(temp,key=lambda x:x[1],reverse=True)  #sorting based on label probability to get top k
        rank1[ labelids.index( temp[0][0] ) / partition_size ] += 1
        rank3[ labelids.index( temp[0][0] ) / partition_size ] += 1
        rank5[ labelids.index( temp[0][0] ) / partition_size ] += 1

        rank3[ labelids.index( temp[1][0] ) / partition_size ] += 1
        rank5[ labelids.index( temp[1][0] ) / partition_size ] += 1
        rank3[ labelids.index( temp[2][0] ) / partition_size ] += 1
        rank5[ labelids.index( temp[2][0] ) / partition_size ] += 1


        rank5[ labelids.index( temp[3][0] ) / partition_size ] += 1
        rank5[ labelids.index( temp[4][0] ) / partition_size ] += 1

    rank1 = [ ( float(x) /testing.totalPages )*100 for x in rank1  ]
    rank3 = [ ( float(x) /( 3 * testing.totalPages) )*100 for x in rank3  ]
    rank5 = [ ( float(x) /( 5 * testing.totalPages) )*100 for x in rank5  ]

    print( rank1)
    print( rank3) 
    print(rank5)

    filePtr = open( outputfile , "w")
    for i in rank1:
        filePtr.write( str(i) + "," )
    filePtr.write("\n")

    for i in rank3:
        filePtr.write( str(i) + "," )
    filePtr.write("\n")

    for i in rank5:
        filePtr.write( str(i) + "," )
    filePtr.close()



if __name__ == '__main__':
    testfile = "../../dataset/minusTop5Labels/wiki10_minusTop5labels_test.txt"
    modelfile = "models_minustop5labels_endon17march_prec@1=0.51/M7_20_10_2-3-4_64_100_100_400_0.001_2/cnn_dynamicmaxpool_100"
    outputfile = "results_minustop5labels_endon17march18_prec0.51/puru_analysis.txt"
    analyse(modelfile,testfile,outputfile)