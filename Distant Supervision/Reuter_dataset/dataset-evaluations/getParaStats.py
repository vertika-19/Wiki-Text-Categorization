import numpy as np 

trainfilename = "../reuters_sparse_training.txt"
inpFile = open(trainfilename,'r')
outputfilename = "train_stats.csv"
outputfile = open(outputfilename,'w')

totalPages = int(inpFile.readline())
outputfile.write(str(totalPages) + "\n")
docid = int(inpFile.readline())
outputfile.write(str(docid)+ "\n")

outputfile.write("id,no of labels,no of para,no of unique words in page,no of words in each para\n")

for count in range(totalPages):
	labels = inpFile.readline().rstrip("\n")
	for i in range(int(labels)):		#just pass 
		x = inpFile.readline().rstrip("\n")
	noofpara = inpFile.readline().rstrip("\n")
	op = str(count) + "," + labels + "," + noofpara+","
	totalwords = 0
	words = set()
	parawords = ""
	for i in range(int(noofpara) ):
		line = inpFile.readline().rstrip("\n").split("\t")
		line = line[1].split()
		line = [int(x.split(":")[0]) for x in line if int(x.split(":")[0]) > 0 ]
		parawords = parawords + str(len(line)) + ","
		words.update(line)

	op = op +  str(len(words)) + "," +  parawords + "\n"
	outputfile.write(op)

                    




