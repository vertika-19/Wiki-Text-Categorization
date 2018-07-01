from DataParser import DataParser as DataParser
from model7 import Model7 as Model
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
    poolLength = int(sys.argv[11])

    labels = 30938
    # labels = 968
    vocabularySize = 101939

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
    output = sys.argv[9]

    for k in K_list:
		precAtK[k] /= (k * len(pred)) 
		print ("Prec@" + str(k) + " = " + str(precAtK[k]))
		output = output + "," + "Prec@" + str(k) + "=," + str(precAtK[k])

    # for key,val in enumerate(precAtK):
    #     if key in K_list :	#As need Prec@k only on Klist ie 1,3,5,10
    #         # print( val ) 
    #         # print( val/len(pred) )
    #         print( "prec@" + key + "=" + str(val/(key*len(pred))) )
    #         # f.write(str(key)+","+ str(val/len(pred))+"\n")
    #         # f.write(str(key)+","+ str(val/(key*len(pred))+"\n")
    #         output = output + ", prec@" + key + "," + str(val/(key*len(pred)))
    f.write(output + "\n")
    f.close()

if __name__ == '__main__':
    # testfile = "../../dataset/minusTop5Labels/wiki10_minusTop5labels_test.txt"
    #testfile = "../../dataset/minusTop5Labels/wiki10_minusTop5labels_test.txt"
    testfile = "/home/khushboo/wiki10/dataset/original_split/wiki10_miml_test.txt"
    # testfile = "/home/khushboo/wiki10/dataset/original_split/wiki10_test_cluster_16.txt"
    
    # modelfile = "/media/hdd1/miml/next-phase/models/" + sys.argv[8] + "/model7_wiki10_" + sys.argv[7]
    # modelfile = "models/" + sys.argv[8] + "/cnn_dynamicmaxpool_parabel_c16_" + sys.argv[7]
    # modelfile = "models/" + sys.argv[8] + "/cnn_dynamicmaxpool_" + sys.argv[7]
    modelfile = "/media/backup/khushboo/next-phase/cnn_dynamic_maxpool_with_dropout/models/" + sys.argv[8] + "/cnn_dynamicmaxpool_" + sys.argv[7]
    
    # outputfile = "results/precAtk.txt"
    outputfile = "results/precAtk_full.txt"
    ComputePrecisionK(modelfile,testfile,outputfile)
