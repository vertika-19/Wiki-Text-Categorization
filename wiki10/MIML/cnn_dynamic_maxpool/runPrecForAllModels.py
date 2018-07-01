import os

#parameters reduced just to compute fscore on khushboo desktop
paragraphLength = [20, 50 , 100] 		
maxParagraphs = [5, 10, 20]				#[10,20,30]
filterSizes = 	["1", "1-2", "2"]				#[ "1", "2-3" , "2-3-4" ]
num_filters = 	[16, 32, 64]						#[ 40,64,80 ]
wordEmbeddingDimension = [30, 50]				#[50]
batchSize = [1, 10, 100]								#[100]
maxepochs = 400									#400
epochs = [100,200,400]								#[100,200,300,400]
learningRate = [1e-2,1e-3]										#[1e-2,1e-3,1e-4,1e-5]

os.system("rm -rf results/precAtk.txt")

for pl in paragraphLength:
	for mp in maxParagraphs:
		for fs in filterSizes:
			for nf in num_filters:
				for wd in wordEmbeddingDimension:
					for bs in batchSize:
						for lrate in learningRate:
							folder = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(maxepochs) + "_" + str(lrate)
							for ep in epochs:
								folder_fscore = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(ep) + "_" + str(lrate)
								os.system("python PrecAtK.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + "  " + str(ep) + "  " + folder + "  " + folder_fscore + " " + str(lrate) )
