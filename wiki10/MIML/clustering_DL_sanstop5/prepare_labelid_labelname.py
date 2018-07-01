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

labelids = open("../../dataset/labelId-labelName-full.txt","r").read().strip().split("\n")
labelids = [  x.split("\t")[1].strip() for x in labelids ]

outFile = open("labelId-labelName-full.txt" ,"w")

for x in range(len(clusters)):
	outFile.write( str(x) + "\t" + labelids[clusters[x]] + "\n" )

outFile.close()
