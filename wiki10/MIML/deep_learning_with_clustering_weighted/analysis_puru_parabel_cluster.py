from DataParser import DataParser as DataParser
from clustering_DL import clustering_DL as Model
import numpy as np
import os, sys
import math

#M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5
#M7_50_10_2-3-4_16_50_100_300_0.001_2_0.8
#M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5

def ComputePrecisionK(modelfile,testfile):
    maxParagraphLength = 15
    maxParagraphs = 10
    filterSizes = [2,3,4]
    num_filters = 16
    wordEmbeddingDimension = 50
    lrate = float(0.001)
    poolLength = 2
    labels = 30938
    vocabularySize = 101939

    keep_prob = 1.0

    noOfLabelsPerClusters = open("cluster_metainfo_wiki10.txt").read().strip().split("\n")
    noOfLabelsPerClusters = [int(x) for x in noOfLabelsPerClusters]
    TotalNoOfInstances = 107016
    noOfInstancePerClusters = open("trn_cluster_noofInstances.txt").read().strip().split("\n")
    noOfInstancePerClusters = [ float(x) / TotalNoOfInstances  for x in noOfInstancePerClusters]

    numberOfClusters = 32
    weightsForCluster = []
    for x in range(numberOfClusters):
        temp = [ noOfInstancePerClusters[x] ] * noOfLabelsPerClusters[x]
        weightsForCluster.extend(temp)


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
        pred.append(list(np.multiply(pre[0], weightsForCluster)))

    # labelids = open("sorted_labelid_dl_clustering.txt","r").read().strip().split("\n")
    # labelids = [ int(x) for x in labelids ]
    
    noOfLabelsPerClusters = open("cluster_metainfo_cumulative.txt").read().strip().split("\n")
    noOfLabelsPerClusters = [int(x) for x in noOfLabelsPerClusters]

    no_of_partition = 32
    partition_size = labels / no_of_partition
    prec1 = [0]*no_of_partition
    prec3 = [0]*no_of_partition
    prec5 = [0]*no_of_partition

    pred1 = [0]*no_of_partition
    pred3 = [0]*no_of_partition
    pred5 = [0]*no_of_partition

    for i,v in enumerate(pred):
        temp = [(labId,labProb) for labId,labProb in enumerate(v) ]
        temp = sorted(temp,key=lambda x:x[1],reverse=True)  #sorting based on label probability to get top k
        #finding how many of these were true
        
        bucket = 31
        for id in range(len(noOfLabelsPerClusters)):
            if temp[0][0] < noOfLabelsPerClusters[id]:
                bucket = id
                break
        pred1[ bucket ] += 1
        pred3[ bucket ] += 1
        pred5[ bucket ] += 1
        if truePre[i][0][temp[0][0]] == 1:
            prec1[ bucket ] += 1
            prec3[ bucket ] += 1
            prec5[ bucket ] += 1

        
        bucket = 31
        for id in range(len(noOfLabelsPerClusters)):
            if temp[1][0] < noOfLabelsPerClusters[id]:
                bucket = id
                break
        pred3[ bucket ] += 1
        pred5[ bucket ] += 1
        if truePre[i][0][temp[1][0]] == 1:
            prec3[ bucket ] += 1
            prec5[ bucket ] += 1

        bucket = 31
        for id in range(len(noOfLabelsPerClusters)):
            if temp[2][0] < noOfLabelsPerClusters[id]:
                bucket = id
                break
        pred3[ bucket ] += 1
        pred5[ bucket ] += 1       
        if truePre[i][0][temp[2][0]] == 1:
            prec3[ bucket ] += 1
            prec5[ bucket ] += 1

        bucket = 31
        for id in range(len(noOfLabelsPerClusters)):
            if temp[3][0] < noOfLabelsPerClusters[id]:
                bucket = id
                break
        pred5[ bucket ] += 1
        if truePre[i][0][temp[3][0]] == 1:
            prec5[ bucket ] += 1
        

        bucket = 31
        for id in range(len(noOfLabelsPerClusters)):
            if temp[4][0] < noOfLabelsPerClusters[id]:
                bucket = id
                break
        pred5[ bucket ] += 1
        if truePre[i][0][temp[4][0]] == 1:
            prec5[ bucket ] += 1

    print( prec1 )
    print( prec3 ) 
    print( prec5 )

    prec1 = [ ( float(x) /testing.totalPages )*100 for x in prec1  ]
    prec3 = [ ( float(x) /( 3 * testing.totalPages) )*100 for x in prec3  ]
    prec5 = [ ( float(x) /( 5 * testing.totalPages) )*100 for x in prec5  ]

    
    print( prec1 )
    print( prec3 ) 
    print( prec5 )


    print( pred1 )
    print( pred3 ) 
    print( pred5 )

    pred1 = [ ( float(x) /testing.totalPages )*100 for x in pred1  ]
    pred3 = [ ( float(x) /( 3 * testing.totalPages) )*100 for x in pred3  ]
    pred5 = [ ( float(x) /( 5 * testing.totalPages) )*100 for x in pred5  ]

    
    print( pred1 )
    print( pred3 ) 
    print( pred5 )


if __name__ == '__main__':
    testfile = "dataset/wiki10_miml_test_dl.txt"
    modelfile = "models/M7_50_10_2-3-4_16_50_100_300_0.001_2_0.8/clustering_DL_100"
    ComputePrecisionK(modelfile,testfile)

