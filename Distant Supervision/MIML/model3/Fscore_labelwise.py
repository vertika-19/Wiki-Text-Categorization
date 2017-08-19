from DataParser import DataParser as DataParser
from model3 import Model3 as Model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib as plt


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
    maxParagraphLength=20
    maxParagraphs=20
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
    return (sum_fscore / float((labels - 1)))


if __name__ == '__main__':
    testfile = "C:/gitrepo/Wiki-Text-Categorization/Distant Supervision/Reuter_dataset/reuters_sparse_testcvmerged.txt"
    modelfile = "models/model3_reuter_100"
    outputfile = "results/fscorelabelwise_100.txt"
    ComputeFscore(modelfile,testfile,outputfile)

# if __name__ == '__main__':
#     testfile = "C:/gitrepo/Wiki-Text-Categorization/Distant Supervision/Reuter_dataset/reuters_sparse_test.txt"
    
#     fscore_epochs =[]
#     startepoch = 50
#     endepoch = 100
#     epochstep = 10
#     for i in range(startepoch,endepoch,epochstep):
#         print(i)
#         modelfile = "models/model3_reuter_" + str(i)
#         outputfile = "results/fscorelabelwise_" + str(i) + ".txt"
#         print(modelfile)
#         print(outputfile)
#         val = ComputeFscore(modelfile,testfile,outputfile)
#         fscore_epochs.append(val)
#         print(val)

#     print(fscore_epochs)
#     epochslist = list(np.arange(startepoch,endepoch,epochstep))
#     plt.plot(epochslist,fscore_epochs)

#     plt.axis([startepoch,endepoch,0,1])
#     plt.xticks(np.arange(startepoch,endepoch, epochstep))
#     plt.yticks(np.arange(0.6,1, 0.05))

#     plt.ylabel('fscore')
#     plt.xlabel('epochs')
#     plt.show()
#     fig = plt.figure()
#     fig.savefig('miml model3_reuter_fscores.png')
