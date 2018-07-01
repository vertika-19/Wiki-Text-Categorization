import os

#M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5


paragraphLength = [50,15] 		
maxParagraphs = [10,20]
filterSizes = 	[ "2-3-4"]
num_filters = 	[ 64]
wordEmbeddingDimension = [50]				
batchSize = [100]								
maxepochs = 300									
epochs = [100,200,300]						
learningRate = [1e-3]		
poolSize = [5,2]
keep_prob = [0.5,0.8]	
			

os.system("rm -rf results" )
os.system("mkdir results")
# os.system("rm -rf models" )
# os.system("mkdir models")

for pl in paragraphLength:
	for mp in maxParagraphs:
		for fs in filterSizes:
			for nf in num_filters:
				for wd in wordEmbeddingDimension:
					for bs in batchSize:
						for lrate in learningRate:
							for poolLength in poolSize:
								for kp in keep_prob:
									folder = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" + str(nf) \
									+ "_" +  str(wd) + "_" + str(bs) + "_" + str(maxepochs) + "_" + str(lrate) \
									+ "_" + str(poolLength) + "_" + str(kp)
									
									# if folder != "M7_50_10_2-3-4_64_50_100_300_0.001_5_0.5":
									# 	continue
										
									os.system("mkdir /media/hdd2/miml/clustering_DL_sanstop5/models/" + folder)
									
									print("LOG: Iteration for params: [ paragraphLength=" \
										+ str(pl) + " maxParagraphs=" + str(mp) + " filterSizes=" \
										+ str(fs) +"  num_filters=" + str(nf)  +" wordEmbeddingDimension=" \
										+ str(wd) + " batchSize=" + str(bs) + " maxepochs=" + str(maxepochs) \
										+ "  foldername=" + folder + " learning rate=" + str(lrate) \
										+ " poolLength=" + str(poolLength) \
										+ " keep_prob=" + str(kp) + " ]" )
									
									os.system("python main.py " + str(pl) + " " + str(mp) + " " \
										+ str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) \
										+ " " + str(maxepochs) + " " + folder + " " + str(lrate) \
										+ " " + str(poolLength) + " " + str(kp) )
									
									# for ep in epochs:
									# 	print("LOG: Calculating precision" )
									# 	folder_fscore = "M7_" + str(pl) + "_" + str(mp) + "_" +  fs + "_" \
									# 	+ str(nf) + "_" +  str(wd) + "_" + str(bs) + "_" + str(ep) + "_" \
									# 	+ str(lrate) + "_" + str(poolLength) + "_" + str(kp)

									# 	os.system("python PrecAtK.py " + str(pl) + " " + str(mp) + " " \
									# 		+ str(fs) +"  " + str(nf)  +" " + str(wd) + " " + str(bs) \
									# 		+ "  " + str(ep) + "  " + folder + "  " + folder_fscore \
									# 		+ " " + str(lrate) + " " + str(poolLength) + " " + str(kp) ) 
									
									# os.system("mv models/" + folder + " /media/hdd2/miml/clustering_DL_sanstop5/models/")
