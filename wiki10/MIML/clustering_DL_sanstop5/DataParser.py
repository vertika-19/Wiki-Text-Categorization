import numpy as np
import random


class DataParser:
    def __init__(self,maxpara,paragraphLength,labels,vocabSize):
        self.data=[]
        #self.paragraphLength=paraLength
        self.labels=labels
        self.maxParagraphs = maxpara
        self.paragraphLength = paragraphLength
        self.vocabSize=vocabSize
        self.counter = 0
        self.totalPages = 1

    def getDataFromfile(self,fname):

        f=open(fname)
        self.data =[]
        self.totalPages = int(f.readline())
        count = 0
        while count < self.totalPages:
            pageId = f.readline()
            count = count + 1
            labelCount = int(f.readline())
            oneHotLabels = [0] * self.labels
            # print(labelCount)
            for i in range(labelCount):
                labelId = int(f.readline())
                oneHotLabels[labelId] = 1
                assert labelId < self.labels

            sectionCount = int(f.readline())
            docFea = []
            docValue = []
            
            for i in range(sectionCount):
                curSection = f.readline().split()
                sectionFeatId = []
                sectionFeatValue  = [] 
                for x in curSection:
                    if int(x) > 0 and int(x) < self.vocabSize :
                        sectionFeatId.append( int( x ) )
                     #   sectionFeatValue.append( int( float(x[1]) ) ) 
                for i in range(len(sectionFeatId),self.paragraphLength):
                    sectionFeatId.append(0)
                 #   sectionFeatValue.append(1)

                docFea.append( sectionFeatId[:self.paragraphLength] ) 
                #docValue.append( sectionFeatValue[:self.paragraphLength] ) 


            for i in range(sectionCount,self.maxParagraphs):
                docFea.append( [0] * self.paragraphLength) 
            for i in range(self.maxParagraphs):
                docValue.append( [1] * self.paragraphLength) 

            self.data.append((oneHotLabels,docFea[:self.maxParagraphs],docValue[:self.maxParagraphs]))
        self.totalPages = len(self.data)
        f.close()
      
    def nextBatch(self):
        self.counter = self.counter % self.totalPages
        data= self.data[self.counter]
        self.counter += 1
        return data

    def nextBatch(self,batchSize):
        self.counter = self.counter % self.totalPages
        #if self.counter == 0:
        #    random.shuffle(self.data)

        labelBatch = []
        feaIDBatch = []
        feaValBatch =[]
        for itr in range(self.maxParagraphs):
            feaIDBatch.append([])
            feaValBatch.append([])

        for i in range(batchSize):
            labelBatch.append(self.data[self.counter][0])
            for itr in range(self.maxParagraphs):
                feaIDBatch[itr].append(self.data[self.counter][1][itr])
                feaValBatch[itr].append(self.data[self.counter][2][itr])
            self.counter += 1
        return (labelBatch,feaIDBatch,feaValBatch)
    
    def restore(self):
        self.counter=0
