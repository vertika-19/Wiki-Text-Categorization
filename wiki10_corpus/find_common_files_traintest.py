import numpy as np

testfile = open("/home/khushboo/wiki10_corpus/test/wiki10-31K_test_map.txt","r")
trainfile = open("/home/khushboo/wiki10_corpus/wiki10-31K_train_map.txt","r")


with testfile as f:
    testlist = f.readlines()

with trainfile as f:
    trainlist = f.readlines()

s2 = set(trainlist)
commonpages= [val for val in testlist if val in s2]

commonpagesfile = open("/home/khushboo/wiki10_corpus/commonfiles_traintest.txt",'w')
for l in commonpages:
	commonpagesfile.write(l)


testfile.close()
trainfile.close()