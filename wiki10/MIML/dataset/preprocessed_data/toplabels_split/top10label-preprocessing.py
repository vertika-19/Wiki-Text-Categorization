import numpy as np 
import os

trainfilename = "preprocessed_data/split/wiki10_miml_train.txt"
toplabelsfile = "top10labels-wiki10.txt"
inpFile = open(trainfilename,'r')
outputfilename = "preprocessed_data/toplabels_split/wiki10-top10labels_train.txt"
outputfile = open(outputfilename,'w')


toplabels = [ int(x) for x in open(toplabelsfile,'r').read().split("\n") if len(x) > 0] 
# print(toplabels)
totalPages = int(inpFile.readline())
# outputfile.write(str(totalPages) + "\n")

newDocCount = 0

for count in range(totalPages):
	op = ""
	docid = int(inpFile.readline())
	op = op + str(docid)+ "\n"
	noOflabels = inpFile.readline().rstrip("\n")
	tempLabel = []
	for i in range(int(noOflabels)):		#just pass 
		x = int(inpFile.readline().rstrip("\n") )
		if x in toplabels:
			tempLabel.append( toplabels.index(x) )

	op = op + str(len(tempLabel)) + "\n"
	for i in range( len(tempLabel)):
		op += str( tempLabel[i]) + "\n"

	noofpara = inpFile.readline().rstrip("\n")
	op = op + noofpara + "\n"
	for i in range(int(noofpara) ):
		line = inpFile.readline()
		op = op + line

	if len(tempLabel) > 0 :
		newDocCount += 1
		outputfile.write(op)
	# print(op)

if newDocCount != totalPages:
	print("Extra work :( total no of docs = " + str(newDocCount) )

