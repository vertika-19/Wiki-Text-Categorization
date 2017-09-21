import os

paragraphLength = [10]
maxParagraphs = [20]
filterSizes = [ "1" ]
num_filters = [ 64 ]
wordEmbeddingDimension = [50]
batchSize = [100]
maxepochs = 50
epochs = [50]

# paragraphLength = [8,10,20]
# maxParagraphs = [10,20,30]
# filterSizes = [ "1", "2-3" , "2-3-4" ]
# num_filters = [ 40,64,80 ]
# wordEmbeddingDimension = [50]
# batchSize = [100]
# maxepochs = 400
# epochs = [100,200,300,400]

os.system("rm -f results/costfile.txt" )
os.system("rm -f results/fscorelabelwise.txt")

for pl in paragraphLength:
	for mp in maxParagraphs:
		for fs in filterSizes:
			for nf in num_filters:
				for wd in wordEmbeddingDimension:
					for bs in batchSize:
						folder = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(maxepochs)
						os.system("mkdir models\\" + folder)
						print("LOG: Iteration for params: [ paragraphLength=" + str(pl) + " maxParagraphs=" + str(mp) + " filterSizes=" + str(fs) +"  num_filters=" + str(nf)  +" wordEmbeddingDimension=" + str(wd) + " batchSize=" + str(bs) + " maxepochs=" + str(maxepochs) + "  foldername=" + folder + " ]" )
						os.system("python main.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + " " + str(maxepochs) + "  " + folder )
						for ep in epochs:
							folder_fscore = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(ep)
							os.system("python Fscore_labelwise.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + "  " + str(ep) + "  " + folder + "  " + folder_fscore )

