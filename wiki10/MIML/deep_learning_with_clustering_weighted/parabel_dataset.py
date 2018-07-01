noOfClusters = 32


file_clusters = open("32cluster_info_wiki10.txt","r")

temp_clusters = file_clusters.read().strip().split("\n")
temp_clusters = [ ( x.split() ) for x in temp_clusters ]

# list of list and inner list has labels for that cluster
clusters = []
for ct in temp_clusters:
	for x in ct:
		clusters.append( int(x) )

file_clusters.close()

trainfilename = "/home/khushboo/wiki10/dataset/original_split/wiki10_miml_test.txt"
inpFile = open(trainfilename,'r')
outFile = open("/home/khushboo/wiki10/dataset/original_split/wiki10_miml_test_dl.txt", 'w')

totalPages = int(inpFile.readline())
# outputfile.write(str(totalPages) + "\n")

newDocCount = 0

for count in range(totalPages):
	op = ""
	outFile.write(inpFile.readline().strip() + "\n")
	noOflabels = inpFile.readline().strip()
	outFile.write(str(noOflabels) + "\n")
	tempLabel = []
	for i in range(int(noOflabels)):		#just pass 
		x = int(inpFile.readline().strip() )
		outFile.write( str(clusters.index(x)) + "\n" )

	noofpara = int(inpFile.readline().strip())
	outFile.write(str(noofpara) + "\n")
	
	for i in range(noofpara):
		outFile.write(inpFile.readline().strip())
		if(count == totalPages-1 and i == noofpara - 1):
			continue
		outFile.write("\n")
