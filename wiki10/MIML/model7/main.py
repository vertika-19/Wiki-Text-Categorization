from DataParser import DataParser as DataParser
from model7 import Model7 as Model
import tensorflow as tf
import numpy as np
import sys


paragraphLength = int(sys.argv[1])
maxParagraphs = int(sys.argv[2] )
filterSizes = [int(i) for i in sys.argv[3].split("-")]
print(filterSizes)
num_filters = int(sys.argv[4])
wordEmbeddingDimension = int(sys.argv[5])
batchSize= int(sys.argv[6])
epochEnd = int(sys.argv[7])
folder_name = sys.argv[8]
lrate = float(sys.argv[9]) 
nlabels = 10
vocabularySize = 101940

training = DataParser(maxParagraphs,paragraphLength,nlabels,vocabularySize)
training.getDataFromfile("../wiki10_miml_dataset/preprocessed_data/toplabels_split/wiki10-top10labels_train.txt")
model = Model(maxParagraphs,paragraphLength,nlabels,vocabularySize,filterSizes,num_filters,wordEmbeddingDimension,lrate)

costfile = open("results/costfile.txt","a")
output = folder_name

epoch=0
# epochEnd=400
costepochs = []

for e in range(epoch,epochEnd):
    
    cost=0

    for itr in range(int(training.totalPages/batchSize)):
        cost += model.train(training.nextBatch(batchSize))

    if training.totalPages % batchSize > 0:    
        cost += model.train(training.nextBatch(training.totalPages % batchSize))

    # print (str(cost/training.totalPages))


    if (e+1)%50 == 0:
        print ('Epoch: ' + str(e+1))
        print (str(cost/training.totalPages))
        costepochs.append(cost/training.totalPages)
        output = output + "," + str(cost/training.totalPages) 
        print ('saving model..')
        model.save("models/" + folder_name + "/model7_reuter_"+str(e+1))


costfile.write(output + "\n")

# print('cost value at every 10th epoch')
# print(costepochs)
# epochslist = list(np.arange(0,epochEnd,10))
# plt.plot(epochslist,costepochs)

# plt.axis([0,epochEnd,0,1])
# plt.xticks(np.arange(40,epochEnd, 10))
# plt.yticks(np.arange(0,0.7, 0.05))

# plt.ylabel('cost')
# plt.xlabel('epochs')
# plt.show()
# #plt.savefig('miml model3_reuter_epochs100.pdf')
