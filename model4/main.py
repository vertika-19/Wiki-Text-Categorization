#from DataParser_siml import DataParser_siml as DataParser
#from model2_siml import Model2_siml as Model
from DataParser import DataParser as DataParser
from model4 import Model4 as Model

'''
batchSize=1	maxParagraphLength=5 maxParagraphs= 5 	.65
batchSize=1	maxParagraphLength=10 maxParagraphs= 5 	epoch 150 .7242
batchSize=1	maxParagraphLength=15 maxParagraphs= 5 	epoch 150 .70
embedding = 30  filter = [2] EPOCH 70 .74
embedding = 20  filter = [2] EPOCH 100 .75
embedding = 20 EPOCh 100 filter [1] num_filters_parargaph 30 = 0.77
embedding = 20 EPOCh 100 filter [1] num_filters_parargaph 50 0.78
embedding = 20 EPOCh 100 filter [1] num_filters_parargaph 60 = 0.80
batchSize=1	maxParagaphLength=10 maxParagraphs= 7	epoch 120 .69

batchSize=10	maxParagraphLength=10 maxParagraphs= 5 	epoch 200 .71
batchSize=50	maxParagraphLength=10 maxParagraphs= 5 	epoch 250 .69






'''
maxParagraphLength=10
maxParagraphs= 8
#nlabels=1001
#vocabularySize=76391
nlabels=8
vocabularySize=244
training = DataParser(maxParagraphLength,maxParagraphs,nlabels,vocabularySize)
#training.getDataFromfile("data/wiki_fea_76390_Label_1000_train")
training.getDataFromfile("/home/khushboo/Desktop/Reuter_dataset/reuters_sparse_training.txt")

model = Model(maxParagraphLength,maxParagraphs,nlabels,vocabularySize)

batchSize=1

epoch=0
epochEnd=100
for e in range(epoch,epochEnd):
    print 'Epoch: ' + str(e+1)
    cost=0
    for itr in range(int(training.totalPages/batchSize)):
        cost+=model.train(training.nextBatch(batchSize))
    print (str(cost/training.totalPages))

    if (e+1)%10 == 0:
        print 'saving model..'
        model.save("models/model4_reuter_"+str(e+1))
