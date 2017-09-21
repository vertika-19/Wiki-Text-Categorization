#from DataParser_siml import DataParser_siml as DataParser
#from model2_siml import Model2_siml as Model
from DataParser import DataParser as DataParser
from OneVsAll import OneVsAll as Model
import sys


nlabels = 8
vocabularySize = 244
regLambda = float(sys.argv[1])
folder_name = sys.argv[4]

training = DataParser(nlabels,vocabularySize)
training.getDataFromfile("../../dataset/reuters_sparse_training.txt")
model = Model(nlabels,vocabularySize,regLambda)
costfile = open("results/costfile.txt","a")
output = folder_name

batchSize = int(sys.argv[2])

epoch=0
epochEnd= int(sys.argv[3])

for e in range(epoch,epochEnd):
    # print('Epoch: ' + str(e+1) )
    cost = 0
    for itr in range(int(training.totalPages/batchSize)):
        cost += model.train(training.nextBatch(batchSize))
    
    if ( e+1)%10 == 0:
        print('Epoch: ' + str(e+1) )
        print (str(cost/training.totalPages))
        output = output + " , " + str(cost/training.totalPages) 
    

    if (e+1)%50 == 0:
        print ('saving model..')
        model.save("models/" + folder_name + "/model8_reuter_" +str(e+1))

costfile.write(output + "\n")