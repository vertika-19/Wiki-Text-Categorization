from DataParser import DataParser as DataParser
from clustering_DL import clustering_DL as Model
import numpy as np
import os, sys
import math

#M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5
#M7_50_10_2-3-4_16_50_100_300_0.001_2_0.8


def ComputePrecisionK(modelfile,testfile):
    maxParagraphLength = 50
    maxParagraphs = 10
    filterSizes = [2,3,4]
    num_filters = 16
    wordEmbeddingDimension = 50
    lrate = float(0.001)
    poolLength = 2
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

    labelids = open("sorted_labelid_dl_clustering.txt","r").read().strip().split("\n")
    labelids = [ int(x) for x in labelids ]

    no_of_partition = 10
    partition_size = labels / no_of_partition
    prec1 = [0]*no_of_partition
    prec3 = [0]*no_of_partition
    prec5 = [0]*no_of_partition

    for i,v in enumerate(pred):
        temp = [(labId,labProb) for labId,labProb in enumerate(v) ]
        temp = sorted(temp,key=lambda x:x[1],reverse=True)  #sorting based on label probability to get top k
        #finding how many of these were true

        if truePre[i][0][temp[0][0]] == 1:
            prec1[ labelids.index( temp[0][0] ) / partition_size ] += 1
            prec3[ labelids.index( temp[0][0] ) / partition_size ] += 1
            prec5[ labelids.index( temp[0][0] ) / partition_size ] += 1

        if truePre[i][0][temp[1][0]] == 1:
            prec3[ labelids.index( temp[1][0] ) / partition_size ] += 1
            prec5[ labelids.index( temp[1][0] ) / partition_size ] += 1

        if truePre[i][0][temp[2][0]] == 1:
            prec3[ labelids.index( temp[2][0] ) / partition_size ] += 1
            prec5[ labelids.index( temp[2][0] ) / partition_size ] += 1

        if truePre[i][0][temp[3][0]] == 1:
            prec5[ labelids.index( temp[3][0] ) / partition_size ] += 1

        if truePre[i][0][temp[4][0]] == 1:
            prec5[ labelids.index( temp[4][0] ) / partition_size ] += 1

    print( prec1 )
    print( prec3 ) 
    print( prec5 )

    prec1 = [ ( float(x) /testing.totalPages )*100 for x in prec1  ]
    prec3 = [ ( float(x) /( 3 * testing.totalPages) )*100 for x in prec3  ]
    prec5 = [ ( float(x) /( 5 * testing.totalPages) )*100 for x in prec5  ]

    
    print( prec1 )
    print( prec3 ) 
    print( prec5 )

if __name__ == '__main__':
    testfile = "dataset/wiki10_miml_test_dl.txt"
    modelfile = "models/M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5/clustering_DL_300"
    ComputePrecisionK(modelfile,testfile)

