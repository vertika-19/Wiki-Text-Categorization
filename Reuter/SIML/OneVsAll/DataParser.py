
# coding: utf-8

# In[ ]:

import numpy as np
import random
class DataParser:
    def __init__(self,labels,vocabSize):
        self.data=[]
        self.labels=labels
        self.vocabSize=vocabSize

    def getDataFromfile(self,fname):
        self.counter =0
        f=open(fname)
        self.data=[]
        self.totalPages = int(f.readline())

        pageId= f.readline()
        count = 0
        while count < self.totalPages:
            count = count+1
            labelCount=int(f.readline())
            labelsTemp=[0]*self.labels
            for i in range(labelCount):
                temp = int(f.readline())
                if temp < self.labels:
                    labelsTemp[temp]=1

            instancesCount=int(f.readline())
            instancesTemp = [0]*self.vocabSize
            
            for i in range(instancesCount):
                for x in (f.readline().split("\t") )[1].split():
                    x = x.split(":")
                    if int(x[0]) > 0 and int(x[0]) < self.vocabSize :
                        instancesTemp[int(x[0])] += int(float(x[1]) )
                
            self.data.append((labelsTemp,instancesTemp))
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
        for i in range(batchSize):
            self.counter %= self.totalPages
            if self.counter == 0:
                random.shuffle(self.data)
            labelBatch.append(self.data[self.counter][0])
            feaBatch.append(self.data[self.counter][1])
            self.counter += 1
        # print(feaBatch)
        return (labelBatch,feaBatch)
    
    def restore(self):
        self.counter=0
