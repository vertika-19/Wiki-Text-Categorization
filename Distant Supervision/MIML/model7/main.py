from DataParser import DataParser as DataParser
from model7 import Model7 as Model
import tensorflow as tf


maxParagraphs = 10 
paragraphLength = 20
nlabels = 8
vocabularySize = 244
training = DataParser(maxParagraphs,paragraphLength,nlabels,vocabularySize)
#training.getDataFromfile("data/wiki_fea_76390_Label_1000_train")
training.getDataFromfile("C:/gitrepo/Wiki-Text-Categorization/Distant Supervision/Reuter_dataset/reuters_sparse_training.txt")

model = Model(maxParagraphs,paragraphLength,nlabels,vocabularySize)

batchSize=100

epoch=0
epochEnd=120
costepochs = []
for e in range(epoch,epochEnd):
    print ('Epoch: ' + str(e+1))
    cost=0

    for itr in range(int(training.totalPages/batchSize)):
        cost += model.train(training.nextBatch(batchSize))
    if training.totalPages % batchSize > 0:    
        cost += model.train(training.nextBatch(training.totalPages % batchSize))

    print (str(cost/training.totalPages))

    if (e+1)%10 == 0:
        costepochs.append(cost/training.totalPages)
        print ('saving model..')
        model.save("models/model7_reuter_"+str(e+1))

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
