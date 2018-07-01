import numpy as np 

outputFile = open("trn_cluster_noofInstances.txt" , "w" )
instance_sets = [set() for _ in xrange(32)]		



file_clusters = open("32cluster_info_wiki10.txt","r")

temp_clusters = file_clusters.read().strip().split("\n")
temp_clusters = [ ( x.split() ) for x in temp_clusters ]

clusters = []
for ct in temp_clusters:
	clusters.append( [ int(x) for x in ct ]  )

file_clusters.close()

trainfilename = "/home/miml/Wiki-Text-Categorization/wiki10/dataset/original_split/wiki10_miml_train.txt"
inpFile = open(trainfilename,'r')

totalPages = int(inpFile.readline())
newDocCount = 0

for count in range(totalPages):
	inpFile.readline().strip()
	noOflabels = inpFile.readline().strip()
	tempLabel = []
	for i in range(int(noOflabels)):		#just pass 
		x = int(inpFile.readline().strip() )
		for ct in range(32):
			if x in clusters[ct]:
				instance_sets[ct].add(count)


	noofpara = int(inpFile.readline().strip())
	
	for i in range(noofpara):
		inpFile.readline().strip()

for w in instance_sets:
	outputFile.write(str(len(w) ) + "\n")
