import os

regLambda = [80,100,150]
batchSize = [1,64,80,100,120]
maxepochs = 400
epochs = [100,200,300,400]

os.system("rm -f results/costfile.txt" )
os.system("rm -f results/fscorelabelwise.txt")
os.system("rm -rf models")

for reg in regLambda:
	for bs in batchSize:
		folder = "M8_" + str(reg) + "_" + str(bs) + "_" + str(maxepochs)
		os.system("mkdir models\\" + folder)
		print("LOG: Iteration for params: [ reg="  + str(reg) + " batchsize="  + str(bs) + " max_epochs=" + str(maxepochs) + "  foldername=" + folder + " ]" )
		os.system("python main.py " + str(reg) + " " + str(bs) + " " + str(maxepochs) + "  " + folder )
		for ep in epochs:
			folder_fscore = "M8_" + str(reg) +  "_" + str(bs) + "_" + str(ep)
			os.system("python Fscore_labelwise.py " + str(reg) + " "  + str(bs) + " " + str(ep) + " " + folder + " " + folder_fscore )

