file = open("wiki10_miml.txt", "r")
total_doc = int(file.readline())
print(total_doc)
train_doc = 13269
test_doc = 6137
#19406 
check = 1
train_file = open("wiki10_miml_train.txt", "w")
train_file.write(str(train_doc) + "\n")
while 1:
	page_id = int(file.readline())
	print(page_id)
	if page_id == train_doc + 1:
		break
	train_file.write(str(page_id) + "\n")
	tot_label = int(file.readline())
	if page_id == 19247:	#Novikov_self-consistency_principle
		tot_label = tot_label - 1
	if page_id == 18553:	#Information_Technology_Infrastructure_Library
		tot_label = 24
	if page_id == 10091 and check == 1:
		check = 0
		tot_label = 19
	if page_id == 13511:	#ISO_216
		tot_label = 25
	if page_id == 18927:	#Language_Integrated_Query
		tot_label = 18
	if page_id == 13425:	#CPU_socket
		tot_label = 26
	train_file.write(str(tot_label) + "\n")
	count = 0
	while count < tot_label:
		train_file.write(file.readline())
		count = count + 1
	tot_section = int(file.readline())
	train_file.write(str(tot_section) + "\n")
	count = 0
	while count < tot_section:
		train_file.write(file.readline())
		count = count + 1

test_file = open("wiki10_miml_test.txt", "w")
test_file.write(str(test_doc) + "\n")
flag = 1
while 1:
	test_file.write(str(page_id) + "\n")
	print(page_id)
	tot_label = int(file.readline())
	if page_id == 18320 and flag == 1:
		flag = 0
		tot_label = 29
	if page_id == 18738 and check == 0:
		check = 1
		tot_label = 26
	test_file.write(str(tot_label) + "\n")
	count = 0
	while count < tot_label:
		test_file.write(file.readline())
		count = count + 1
	tot_section = int(file.readline())
	test_file.write(str(tot_section) + "\n")
	count = 0
	while count < tot_section:
		test_file.write(file.readline())
		count = count + 1
	if page_id == total_doc:
		break
	page_id = int(file.readline())
	
