import numpy as np 

filename = "preprocessed_data/split/wiki10_miml.txt"
inpFile = open(filename,'r')
outputfilename = "wikilabel_stats.csv"
outputfile = open(outputfilename,'w')

totalPages = int(inpFile.readline())
#outputfile.write(str(totalPages) + "\n")
#outputfile.write("id,no of labels,no of para,no of unique words in page,no of words in each para\n")

labeldict = np.zeros(30938)
for count in range(totalPages):
	docid = int(inpFile.readline())
	print(str(docid))
	labels = inpFile.readline().rstrip("\n")
	for i in range(int(labels)):		#just pass 
		x = inpFile.readline().rstrip("\n")
		labeldict[int(x)] += 1
	noofpara = inpFile.readline().rstrip("\n")
	for i in range(int(noofpara) ):
		line = inpFile.readline().rstrip("\n").split("\t")
		
for i in range(len(labeldict)):
	outputfile.write( str(i) + "," + str(labeldict[i]) + "\n" )
