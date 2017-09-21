import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

trainfilename = "../reuters_sparse_training.txt"
testfilename = "../reuters_sparse_testcvmerged.txt"
inpFile = open(trainfilename,'r')

totalPages = int(inpFile.readline())
docid = int(inpFile.readline())

labelFreq = [0]*8
wordFreq = [0]*244
paraFreq = [0]*2000

docindex  = 0;
for i in range(totalPages):
	labels = inpFile.readline().rstrip("\n")
	for i in range(int(labels)):		#just pass 
		x = int(inpFile.readline().rstrip("\n") )
		labelFreq[x] += 1
	noofpara = int(inpFile.readline().rstrip("\n"))
	paraFreq[docindex] = noofpara
	docindex += 1 
	
	for i in range(noofpara ):
		line = (inpFile.readline().rstrip("\n").split("\t"))[1].split()
		line = [int(x.split(":")[0]) for x in line if int(x.split(":")[0]) > 0 ]
		for word in line:
			wordFreq[word] += 1

inpFile.close()

inpFile = open(testfilename,'r')

totalPages = int(inpFile.readline())
docid = int(inpFile.readline())
for i in range(totalPages):
	labels = inpFile.readline().rstrip("\n")
	for i in range(int(labels)):		#just pass 
		x = int(inpFile.readline().rstrip("\n"))
		labelFreq[x] += 1
	noofpara = int(inpFile.readline().rstrip("\n"))
	paraFreq[docindex] = noofpara
	docindex += 1 
	
	for i in range(noofpara ):
		line = (inpFile.readline().rstrip("\n").split("\t"))[1].split()
		line = [int(x.split(":")[0]) for x in line if int(x.split(":")[0]) > 0 ]
		for word in line:
			wordFreq[word] += 1
inpFile.close()

# plt.xlabel('Label Id')
# plt.ylabel('No of documents')
# plt.axis( [1,7,0,900] )
# patch = mpatches.Patch(color='blue', label='No of documents having a particular label')
# plt.legend(handles=[patch])
# plt.plot( labelFreq )
# plt.show()

# plt.xlabel('Term Id')
# plt.ylabel('Occurrence Frequency in corpus')
# plt.axis( [1,244,max(min(wordFreq)-1,0),max(wordFreq)+1] )
# patch = mpatches.Patch(color='blue', label='No of times a term appears in corpus')
# plt.legend(handles=[patch])
# plt.plot( wordFreq )
# plt.show()

plt.xlabel('Document Id')
plt.ylabel('No of paragraphs')
plt.axis( [1,2005,max(min(paraFreq)-1,0),max(paraFreq)+5] )
patch = mpatches.Patch(color='blue', label='No of paragraphs in each document')
plt.legend(handles=[patch])
plt.plot( paraFreq )
plt.show()