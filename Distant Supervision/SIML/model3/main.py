#from DataParser_siml import DataParser_siml as DataParser
#from model2_siml import Model2_siml as Model
from DataParser import DataParser as DataParser
from model3 import Model3 as Model


maxParagraphLength=200
maxParagraphs= 1
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
for e in range(epoch,epochEnd):
    print('Epoch: ' + str(e+1) )
    cost=0
    for itr in range(int(training.totalPages/batchSize)):
        cost+=model.train(training.nextBatch(batchSize))
    print (str(cost/training.totalPages))

    if (e+1)%10 == 0 and e > 50:
        print ('saving model..')
        model.save("models/model3_reuter_"+str(e+1))