from DataParser import DataParser as DataParser
from model7 import Model7 as Model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib as plt
import sys


def thresholdTuning(tr,pr):
    pre = set(pr)
    pre=set([round(elem,4) for elem in pre])
    bestF=0
    bestThre=0
    pr=np.array(pr)
    for thre in pre:
        scr=f1_score(tr,pr>=thre)
        if scr>bestF:
            bestF=scr
            bestThre=thre
    return bestF,bestThre

 
def genAnalysis(modelfile,testfile,confusionFile):
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

    testing.restore()
    truePre=[]
    pred=[]
    for itr in range(testing.totalPages):
        data=testing.nextBatch(1)
        truePre.append(data[0])
        pre=model.predict(data)
        pred.append(pre[0])

    valid=int(len(truePre)*0.5) #using first 25% data for threshold tuning - we have merged test and cv files
    thresLab={}
    for la in range(labels):
        t=[]
        p=[]
        for i in range(valid):
            t.append(truePre[i][0][la])
            p.append(pred[i][la])
        bestF,bestThre=thresholdTuning(t,p)
        thresLab[la]=bestThre

    print( thresLab)

    labelIDName = open("../labelId-labelName-full.txt").read().split("\n")
    labelIDName = [  [ int(x.split("\t")[0]) , x.split("\t")[1].rstrip() ] for x in labelIDName]
    # print(labelIDName)    

    #making it a dictionary
    labelname = dict(labelIDName)
    # print(labelName[9026])

    f = open(confusionFile,"w")
    for itr in range(valid,testing.totalPages):   #on next 75% getting analaysis
        predLabel = [ pred[itr][i] > thresLab[i] for i in range(labels) ]
        output = ""
        for i in range(labels):
            if predLabel[i] == 1:
                output = output + "," + labelname[i]

        tn, fp, fn, tp = confusion_matrix(truePre[itr][0],predLabel).ravel()
        f.write(str(itr) + "," + str(tn) + "," + str(fp) + "," + str(fn) + "," + str(tp) +  "," + output + "\n")
    f.close()


if __name__ == '__main__':
    testfile = "../wiki10_miml_test.txt"
    modelfile = "/home/khushboo/Desktop/MTP/temp_model7/M7_20_5_1_64_30_1_400_0.01/model7_wiki10_100"
    outputfile = "results/thresholded_analysis.txt"
    genAnalysis(modelfile,testfile,outputfile)
    