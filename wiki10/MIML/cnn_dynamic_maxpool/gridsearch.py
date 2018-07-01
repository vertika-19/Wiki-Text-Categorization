import os

paragraphLength = [20,15,50] 		
maxParagraphs = [10,20]
filterSizes = 	[ "2-3-4"]
num_filters = 	[ 16,64]
wordEmbeddingDimension = [50]				
batchSize = [100]								
maxepochs = 400									
epochs = [100,200,300,400]						
learningRate = [1e-3]		
poolSize = [2,5]					

os.system("rm -rf results" )
os.system("mkdir results")
os.system("rm -rf models" )
os.system("mkdir models")

for pl in paragraphLength:
	for mp in maxParagraphs:
		for fs in filterSizes:
			for nf in num_filters:
				for wd in wordEmbeddingDimension:
					for bs in batchSize:
						for lrate in learningRate:
							for poolLength in poolSize:								
								folder = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) \
								+ "_" +  str(wd) + "_" + str(bs) + "_" + str(maxepochs) + "_" + str(lrate) \
								+ "_" + str(poolLength)
								
								os.system("mkdir models/" + folder)
								
								print("LOG: Iteration for params: [ paragraphLength=" \
									+ str(pl) + " maxParagraphs=" + str(mp) + " filterSizes=" \
									+ str(fs) +"  num_filters=" + str(nf)  +" wordEmbeddingDimension=" \
									+ str(wd) + " batchSize=" + str(bs) + " maxepochs=" + str(maxepochs) \
									+ "  foldername=" + folder + " learning rate=" + str(lrate) \
									+ " poolLength=" + str(poolLength) + " ]" )
								
								os.system("python main.py " + str(pl) + " " + str(mp) + " " \
									+ str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) \
									+ " " + str(maxepochs) + "  " + folder + " " + str(lrate) \
									+ " " + str(poolLength) )
								
								for ep in epochs:
									print("LOG: Calculating precision" )
									folder_fscore = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" \
									+ str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(ep) + "_" \
									+ str(lrate) + "_" + str(poolLength) 

									os.system("python PrecAtK.py " + str(pl) + " " + str(mp) + " " \
										+ str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) \
										+ "  " + str(ep) + "  " + folder + "  " + folder_fscore \
										+ " " + str(lrate) + " " + str(poolLength)  ) 
								
								os.system("mv models/" + folder + " /media/backup/khushboo/next-phase/models/")
