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

 
def ComputeFscore(modelfile,testfile,outputfile):
    maxParagraphLength = int(sys.argv[1])
    maxParagraphs = int(sys.argv[2] )
    filterSizes = [int(i) for i in sys.argv[3].split("-")]
    num_filters = int(sys.argv[4])
    wordEmbeddingDimension = int(sys.argv[5])
    # batchSize= int(sys.argv[6])
    # epochs= int(sys.argv[7])
    # folder_name = sys.argv[8]
    # output = sys.argv[9]
    lrate = sys.argv[10]

    labels = 8
    vocabularySize = 244

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

    labelsCount={}
    ConfusionMa={}
    fScr={}

    thres=0.5
    valid=int(len(truePre)*0.5) #using first 50% data for threshold tuning - we have merged test and cv files
    labelsCount={}
    ConfusionMa={}
    fScr={}
    thresLab={}
    for la in range(labels):
        if la%25==0:
            print("Current label",la)
        t=[]
        p=[]
        for i in range(valid):
            t.append(truePre[i][0][la])
            p.append(pred[i][la])
        bestF,bestThre=thresholdTuning(t,p)
    
        t=[]
        p=[]
        for i in range(valid,len(truePre)):
            t.append(truePre[i][0][la])
            p.append(pred[i][la])
    
        p=np.array(p)
        fScr[la]=f1_score(t,p>=bestThre)
        ConfusionMa[la]= confusion_matrix(t,p>bestThre)
        thresLab[la]=bestThre
    
    f=open(outputfile,"a")
    output = sys.argv[9]
    
    sum_fscore = 0.0
    for i in range(labels):
        sum_fscore = sum_fscore + fScr[i]
        output = output + "," + str(fScr[i])
    output += "," + str(sum_fscore / float(labels - 1))
    print("Fscore at " + sys.argv[7] + " epochs: " + str(sum_fscore / float(labels - 1)) )
    f.write(output + "\n")
    f.close()


if __name__ == '__main__':
    testfile = "../../Reuter_dataset/reuters_sparse_testcvmerged.txt"
    modelfile = "models/" + sys.argv[8] + "/model7_reuter_" + sys.argv[7]
    outputfile = "results/fscorelabelwise.txt"
    ComputeFscore(modelfile,testfile,outputfile)