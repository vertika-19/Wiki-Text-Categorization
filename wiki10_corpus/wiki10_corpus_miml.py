import re
import urllib
from bs4 import BeautifulSoup
 
#extracting labels from standard wiki10 train file for a page
labelfile = open("wiki10_siml_train.txt", "r")
labelfile.readline()

#wikepedia page names with space replaced by underscore - from std dataset
with open('wiki10-31K_train_map.txt') as page_list:
	pages = page_list.read().splitlines()

'''
writing in a file for format suitabel for preprocessing
'''

#counter for pages
pageid = 0
pageMap = open("wiki10_corpus_miml_train.txt", 'w')
delPages = open("wiki10_delPages.txt", 'w')

for pagename in pages:
	
	#getting wiki page content in xml format by name
	html = urllib.urlopen('https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=extracts&exlimit=1&explaintext&titles=' + pagename)
	soup = BeautifulSoup(html, "lxml")
	page = soup.get_text().encode('utf-8')

	#first list is label , second list was features
	labels = labelfile.readline().split()[0]
	labels = re.sub(',', '\n', labels)	#in origial dataset labels were separated by comma
	
	sections = page.split('\n\n\n')
	if len(sections[0].strip()) == 0: 
		delPages.write(str(pageid) + "," + pagename + "\n")
		continue


	pageMap.write("####Page Name####\n")
	pageMap.write( pagename + "\n")
	print "PageNum: " + str(pageid)
	pageid = pageid + 1
	
	pageMap.write( "#####Categories#####\n")
	pageMap.write( labels + "\n")
	

	pageMap.write( "######Section Name######\n")
	pageMap.write( "Introduction\n")
	pageMap.write( sections[0].strip() + "\n")	#first section doesn't have a section name in dataset so writing directly
	for i in range(1,len(sections)):	#for all sections
		'''section heading is like == section name == 
		This was done so that all sections have same no. of == in beg and end of section name
		'''
		sections[i] = re.sub("===+", "==", sections[i])
		para = sections[i].split("==")
		heading = para[1].strip()	#sec name
		content = para[2].strip()	#sec content - consists of all para of a section
		if heading == "See also" or heading == "References":	#ignoring sections after this
			break
		if len(content) != 0:
			pageMap.write( "######Section Name######\n")
			pageMap.write( heading + "\n")
			content = re.sub("\n\n+", "\n", content)
			pageMap.write( content +"\n")
	if i != len(sections) - 1:
		pageMap.write("\n")
