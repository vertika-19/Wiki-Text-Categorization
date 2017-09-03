#from DataParser_siml import DataParser_siml as DataParser
#from model2_siml import Model2_siml as Model
from DataParser import DataParser as DataParser
from model3 import Model3 as Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


'''
maxParagraphLength=15
maxParagraphs= 7
														Avf Macro Fscore
batchSize=10	maxParagraphLength=15 maxParagraphs= 7	.806/0.7906
batchSize=40	maxParagraphLength=15 maxParagraphs= 7 	.79

batchSize=128	maxParagraphLength=15 maxParagraphs= 7 	.804
batchSize=128	maxParagraphLength=15 maxParagraphs= 7	.784

batchSize=50	maxParagraphLength=15 maxParagraphs= 7 	epoch 90	0.8173
batchSize=50	maxParagraphLength=15 maxParagraphs= 7 	epoch 140	0.8139

batchSize=50	maxParagraphLength=10 maxParagraphs= 7 	epoch 140	0.8139
batchSize=50	maxParagraphLength=10 maxParagraphs= 7 	epoch 140	0.8139

batchSize=50	maxParagraphLength=20 maxParagraphs= 7	epoch 90/100/120/140 approx 0.778


batchSize=80	maxParagraphLength=15 maxParagraphs= 7	epoch 90	0.79149
batchSize=80	maxParagraphLength=15 maxParagraphs= 7	eoch 100	0.80776

batchSize=80	maxParagraphLength=20 maxParagraphs= 7	epoch 100	0.8292
batchSize=80	maxParagraphLength=20 maxParagraphs= 7	epoch 90	0.82564

batchSize=100	maxParagraphLength=15 maxParagraphs= 7	0.8019
batchSize=100	maxParagraphLength=15 maxParagraphs= 10	 0.79976

batchSize=100	maxParagraphLength=20 maxParagraphs= 7	 0.81/0.78725
batchSize=100	maxParagraphLength=20 maxParagraphs= 10	 0.82/0.79898


batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 8  0.81252

batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 10  0.84438
batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 16	0.8291
batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 10 filters 64	0.855049


batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 10 filters=64 filterSizes_paragraph = [2,3]	0.866031

batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 10 filters=64 filterSizes_paragraph = [2,3,4]	0.849482
batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 5 filters=64 filterSizes_paragraph = [2,3]	0.85099


batchSize=100	maxParagraphLength=20 maxParagraphs= 10	poolLength= 10 filters=64 filterSizes_paragraph = [2]	0.852203


On train and cv merged but using original test data in some way
batchSize=100	maxParagraphLength=10 maxParagraphs= 10	poolLength= 5 filters=64 filterSizes_paragraph = [1]	0.820137409822 


'''


maxParagraphLength=20
maxParagraphs= 10 #20
#nlabels=1001
#vocabularySize=76391
nlabels=8
vocabularySize=244
training = DataParser(maxParagraphLength,maxParagraphs,nlabels,vocabularySize)
#training.getDataFromfile("data/wiki_fea_76390_Label_1000_train")
training.getDataFromfile("C:/gitrepo/Wiki-Text-Categorization/Distant Supervision/Reuter_dataset/reuters_sparse_training.txt")

model = Model(maxParagraphLength,maxParagraphs,nlabels,vocabularySize)

batchSize=100

epoch=0
epochEnd=120
costepochs = []
for e in range(epoch,epochEnd):
    print ('Epoch: ' + str(e+1))
    cost=0

    for itr in range(int(training.totalPages/batchSize)):
        cost+=model.train(training.nextBatch(batchSize))
        #model.writer.add_summary(model.summary,epoch)
        #model.writer.flush()
    
    if (e+1)%10 == 0:
    	costepochs.append(cost/training.totalPages)
    print (str(cost/training.totalPages))

    if (e+1)%10 == 0 and e > 50:
        print ('saving model..')
        model.save("models/model3_reuter_"+str(e+1))

print('cost value at every 10th epoch')
print(costepochs)
epochslist = list(np.arange(0,epochEnd,10))
plt.plot(epochslist,costepochs)

plt.axis([0,epochEnd,0,1])
plt.xticks(np.arange(40,epochEnd, 10))
plt.yticks(np.arange(0,0.7, 0.05))

plt.ylabel('cost')
plt.xlabel('epochs')
plt.show()
#plt.savefig('miml model3_reuter_epochs100.pdf')
