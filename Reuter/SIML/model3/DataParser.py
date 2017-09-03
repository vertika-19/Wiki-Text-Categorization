
# coding: utf-8

# In[ ]:

import numpy as np
import random
class DataParser:
    def __init__(self,paraLength,maxPara,labels,vocabSize):
        self.data=[]
        self.paragraphLength=paraLength
        self.maxParagraph=maxPara
        self.labels=labels
        self.vocabSize=vocabSize

    def getDataFromfile(self,fname):
        self.counter =0
        self.totalPages=0
        f=open(fname)
        self.data=[]
        totalPages = int(f.readline())
        count=0
        maxWordsInParagraph=self.paragraphLength
        maxParagraphs=self.maxParagraph
        totalLabels=self.labels

        dummyParagraph =[0]*maxWordsInParagraph

        pageId= f.readline()
        count = 0
        while count < totalPages:
            count = count+1
            labelCount=int(f.readline())
            labelsTemp=[0]*totalLabels
            for i in range(labelCount):
                tempLab=int(f.readline())
                labelsTemp[tempLab]=1
                assert tempLab<totalLabels
            instancesCount=int(f.readline())

            if instancesCount > 0:
                instancesTemp = []
                for i in range(instancesCount):
                    tempInstance=f.readline().split("\t")
                    tempInstance = tempInstance[1].split()
                    temp=[int(x.split(":")[0]) for x in tempInstance if int(x.split(":")[0]) > 0 and int(x.split(":")[0]) < self.vocabSize]
                    instancesTemp.extend(temp)

                #to get onlu uninque values among all para
                instancesTemp = list(set(instancesTemp))
                for i in range(len(instancesTemp),maxWordsInParagraph):
                    instancesTemp.append(0)  

                #doing truncations to match limits    
                instancesTemp[:maxWordsInParagraph]
                tempPara = []
                tempPara.append(instancesTemp)
                self.data.append((labelsTemp,tempPara[:maxParagraphs]))
            else:
                self.data.append((labelsTemp,dummyParagraph))
        self.totalPages = len(self.data)
        f.close()
        
          
        
    def nextBatch(self):
        if self.counter >=self.totalPages:
            self.counter=0
        data= self.data[self.counter]
        self.counter+=1
        return data

    def nextBatch(self,batchSize):
        self.counter %= self.totalPages
        if self.counter == 0:
            random.shuffle(self.data)
        labelBatch=[]
        feaBatch=[]
        for itr in range(self.maxParagraph):
            feaBatch.append([])
        for i in range(batchSize):
            self.counter %= self.totalPages
            if self.counter == 0:
                random.shuffle(self.data)
            labelBatch.append(self.data[self.counter][0])
            for itr in range(self.maxParagraph):
                feaBatch[itr].append(self.data[self.counter][1][itr])
            self.counter+=1
        return (labelBatch,feaBatch)
    
    def restore(self):
        self.counter=0
