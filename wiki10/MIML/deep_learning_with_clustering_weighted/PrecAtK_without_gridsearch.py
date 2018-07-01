from DataParser import DataParser as DataParser
from model7 import Model7 as Model
import numpy as np
import os, sys
import math

# M7_20_10_2-3_16_50_100_400_0.001_5

def ComputePrecisionK(modelfile,testfile, param , ep, outputfile):
    
    param = param.split("_")
    
    maxParagraphLength = param[1]
    maxParagraphs = param[2]
    filterSizes = [ int(x) for x in param[3].split("-") ]

    num_filters = param[4]
    wordEmbeddingDimension = param[5]
    lrate = float(param[8])
    poolLength = param[9]
    labels = 30938
    vocabularySize = 101939


    # print(filterSizes)
    # print(maxParagraphLength)
    # print(maxParagraphs)
    # print(lrate)

    model = Model(maxParagraphs,maxParagraphLength,labels,vocabularySize,filterSizes,num_filters,wordEmbeddingDimension,lrate, poolLength)

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
    output = str(m) + "/cnn_dynamicmaxpool_" + str(ep)

    for k in K_list:
		precAtK[k] /= (k * len(pred)) 
		print ("Prec@" + str(k) + " = " + str(precAtK[k]))
		# output = output + "," + "Prec@" + str(k) + "=," + str(precAtK[k])

    f.write(output + "\n")
    f.close()

if __name__ == '__main__':
    testfile = "/home/khushboo/wiki10/dataset/original_split/wiki10_miml_test.txt"

    models = [        "M7_20_20_2-3_16_50_100_400_0.001_2" , "M7_20_10_2-3_16_30_100_400_0.001_5" \
                    , "M7_20_20_2-3_16_50_100_400_0.001_5" , "M7_20_20_2-3_64_30_100_400_0.001_2" \
                    , "M7_20_10_2-3_64_30_100_400_0.001_2" , "M7_20_20_2-3_64_30_100_400_0.001_5" \
                    , "M7_20_10_2-3_64_30_100_400_0.001_5" , "M7_20_20_2-3_64_50_100_400_0.001_2" \
                    , "M7_20_10_2-3_64_50_100_400_0.001_2" , "M7_20_20_2-3_64_50_100_400_0.001_5" \
                    , "M7_20_10_2-3_64_50_100_400_0.001_5" , "M7_20_10_2-3_16_50_100_400_0.001_5" \
                    , "M7_20_20_2-3_16_30_100_400_0.001_5" ]
    epochs = [100,200,300,400]

    for m in models:
        for ep in epochs:
            print( m + "/cnn_dynamicmaxpool_" + str(ep)  )
        modelfile = "/media/backup/khushboo/graminvani_backup/next_phase/models_maxpool_firstrun/" + m + "/cnn_dynamicmaxpool_" + str(ep)
        outputfile = "results/precAtk_to_find_correct_model_for_prec1_0.7702.txt"
        ComputePrecisionK(modelfile,testfile, m, ep , outputfile)


        # "/media/backup/khushboo/graminvani_backup/next_phase/models_maxpool_firstrun/M7_20_10_2-3_16_50_100_400_0.001_5/cnn_dynamicmaxpool_100"
