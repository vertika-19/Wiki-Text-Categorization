from DataParser import DataParser as DataParser
from model3 import Model3 as Model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np


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
    maxParagraphLength=100
    maxParagraphs=1
    #nlabels=1001
    #vocabularySize=76391
    labels=8
    vocabularySize=244
    model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)

    testing = DataParser(maxParagraphLength,maxParagraphs,labels,vocabularySize)
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
    valid=int(len(truePre)*0.35)
    labelsCount={}
    ConfusionMa={}
    fScr={}
    thresLab={}
    for la in range(labels):
        if la%25==0:
            print("Currnet label",la)
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
    
    f=open(outputfile,"w")
    sum_fscore = 0.0
    for i in range(labels):

        sum_fscore = sum_fscore + fScr[i]
        inp=str(i)+","+str(thresLab[i])+","+str(fScr[i])+"\n"
        f.write(inp)
    f.write(str(sum_fscore / float(labels - 1)))

    print(sum_fscore)
    print(sum_fscore / float((labels - 1)))
    f.close()


if __name__ == '__main__':
    modelfile = "models/model3_reuter_100"
    testfile = "C:/gitrepo/Wiki-Text-Categorization/Distant Supervision/Reuter_dataset/reuters_sparse_testcvmerged.txt"
    outputfile = "results/fscorelabelwise_14.txt"

    ComputeFscore(modelfile,testfile,outputfile)
