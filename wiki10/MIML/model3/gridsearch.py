import os

paragraphLength = [80,100,120] 		
maxParagraphs = [10,20]
filterSizes = 	[ "1", "2-3" , "2-3-4" ]
num_filters = 	[ 40,64,80 ]
wordEmbeddingDimension = [30,50]				#[50]
batchSize = [1,10,100]								#[100]
maxepochs = 350									#400
epochs = [100,200,300]								#[100,200,300,400]
learningRate = [1e-2,1e-3,1e-4]										#[1e-2,1e-3,1e-4,1e-5]
poolLength = [5,10,15]

os.system("rm -f results/costfile.txt" )
os.system("rm -f results/fscorelabelwise.txt")

for pl in paragraphLength:
	for mp in maxParagraphs:
		for fs in filterSizes:
			for nf in num_filters:
				for wd in wordEmbeddingDimension:
					for bs in batchSize:
						for lrate in learningRate:
							for pool in poolLength:
								folder = "M3_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(maxepochs) + "_" + str(lrate) + "_" + str(pool)
								os.system("mkdir models\\" + folder)
								print("LOG: Iteration for params: [ paragraphLength=" + str(pl) + " maxParagraphs=" + str(mp) + " filterSizes=" + str(fs) +"  num_filters=" + str(nf)  +" wordEmbeddingDimension=" + str(wd) + " batchSize=" + str(bs) + " maxepochs=" + str(maxepochs) + "  foldername=" + folder + " learning rate=" + str(lrate) + " poolLength=" + str(pool) + " ]" )
								os.system("python main.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + " " + str(maxepochs) + "  " + folder + " " + str(lrate) + " " + str(pool) )
								for ep in epochs:
									folder_fscore = "M3_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(ep) + "_" + str(lrate) + "_" + str(pool)
									os.system("python Fscore_labelwise.py " + str(pl) + " " + str(mp) + " " + str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) + "  " + str(ep) + "  " + folder + "  " + folder_fscore + " " + str(lrate) + " " + str(pool) )

