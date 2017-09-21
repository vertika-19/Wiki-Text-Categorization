import os

paragraphLength = [20] 		
maxParagraphs = [10]				#[10,20,30]
filterSizes = 	["2-3"]				#[ "1", "2-3" , "2-3-4" ]
num_filters = 	[64]						#[ 40,64,80 ]
wordEmbeddingDimension = [50]				#[50]
batchSize = [10]								#[100]
maxepochs = 500									#400
epochs = [100]								#[100,200,300,400]
learningRate = [1e-3]										#[1e-2,1e-3,1e-4,1e-5]

os.system("rm -f results/costfile.txt" )
os.system("rm -f results/fscorelabelwise.txt")

for pl in paragraphLength:
	for mp in maxParagraphs:
		for fs in filterSizes:
			for nf in num_filters:
				for wd in wordEmbeddingDimension:
					for bs in batchSize:
						for lrate in learningRate:
							folder = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(maxepochs) + "_" + str(lrate)
							os.system("mkdir models\\" + folder)
							print("LOG: Iteration for params: [ paragraphLength=" + str(pl) + " maxParagraphs=" + str(mp) + " filterSizes=" + str(fs) +"  num_filters=" + str(nf)  +" wordEmbeddingDimension=" + str(wd) + " batchSize=" + str(bs) + " maxepochs=" + str(maxepochs) + "  foldername=" + folder + " learning rate=" + str(lrate) + " ]" )
							os.system("python main.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + " " + str(maxepochs) + "  " + folder + " " + str(lrate) )
							for ep in epochs:
								folder_fscore = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(ep) + "_" + str(lrate)
								# os.system("python Fscore_labelwise.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + "  " + str(ep) + "  " + folder + "  " + folder_fscore + " " + str(lrate) )

