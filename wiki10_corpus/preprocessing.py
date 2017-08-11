import re
import sklearn
import operator
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer

'''
Note : All paragraphs in a section are merged into one after preprocessing
'''

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def getWordID(sectionData, wordDict, outputFile):
    if len(sectionData.rstrip()) == 0:
        return
    words = sectionData.split()
    for w in words:
        cleanWord = clean_str(w)
        if cleanWord in wordDict:
            outputFile.write(str(wordDict[cleanWord]) + " ")    
        else:
            outputFile.write(str(0) + " ")
            
    outputFile.write("\n")

stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
stopwords_filtered = list(stopwords_nltk.difference(relevant_words))


#corpus file which has all data in text
inputFileName = "/home/khushboo/wiki10_corpus/wiki10_compLeteCorpus_miml.txt"

#Set of files to be opened
inputFile = open(inputFileName)
pageMappingFile = open("/home/khushboo/wiki10_corpus/preprocessing/PageName2PageIdMap.txt", 'w')
categoryMappingFile = open("/home/khushboo/wiki10_corpus/preprocessing/CategoryName2CategoryIdMap.txt", 'w')
outputFile = open("/home/khushboo/wiki10_corpus/preprocessing/wiki10_miml.txt", 'w')
vocabSizeFile = open("/home/khushboo/wiki10_corpus/preprocessing/vocabSize.txt", 'w')

print "Staring the initial parsing..."

#set of data structures to store mappings and content
pageDict = {}       #page name -> index of a page (id)
pageSecCntDict = {}         #page name ->  no of sections in a page
categoryDict = {}       #indexing for each unique category in dataset -> id
wordDict = {}           #word -> id
pageCategCntDict = {}       #page name -> no of categories/labels
pageName = ""
corpus = []

pageCnt = 0         #no of pages in dataset
categoryCnt = 1     #indexing for each unique category in dataset - count of all labels in dataset
noOfSecPerPage = 0      #no of sections in a page
firstPage = 0   
noOfCategPerPage = 0
sectionText = ""    #all para of a section appended together - text
firstSec = 0

ite = 0
donePrint = 1
typeOfLineReadFromFile = 0

for line in inputFile:
    
    if ite % 50 == 0 and donePrint == 1:
        print "Initial parsing: " + str(ite) + " pages parsed..."
    donePrint = 0
    
    if line == "####Page Name####\n":
        ite = ite + 1
        donePrint = 1
        firstSec = 0
        if firstPage == 1:
            if len(sectionText.rstrip()) == 0:
                noOfSecPerPage = noOfSecPerPage - 1
            pageSecCntDict[pageName] = noOfSecPerPage

        firstPage = 1
        noOfSecPerPage = 0
        typeOfLineReadFromFile = 1
        noOfCategPerPage = 0
        continue
    elif line == "######Section Name######\n":
        if runningCat == 1:
            runningCat = 0
            pageCategCntDict[pageName] = noOfCategPerPage
            
        typeOfLineReadFromFile = 2
        continue
    elif line == "#####Categories#####\n":
        typeOfLineReadFromFile = 3
        continue  
    
    if typeOfLineReadFromFile == 0:     #"Normal Text"
        corpus.append(clean_str(line))
        sectionText = sectionText + line
    
    elif typeOfLineReadFromFile == 1:   #page name
        #print "New Page: " + line.rstrip()
        pageName = line.rstrip()
        pageCnt = pageCnt + 1
        #print pageCnt
        pageDict[line.rstrip()] = pageCnt
        
        typeOfLineReadFromFile = 0
        
    elif typeOfLineReadFromFile == 2:
        #print "New section: " + line.rstrip()
        if len(sectionText.rstrip()) == 0 and firstSec == 1:
            sectionText = ""
            typeOfLineReadFromFile = 0
            firstSec = 1
            continue
        sectionText = ""
        typeOfLineReadFromFile = 0
        noOfSecPerPage = noOfSecPerPage + 1
        firstSec = 1
        
    elif typeOfLineReadFromFile == 3:
        noOfCategPerPage = noOfCategPerPage + 1
        runningCat = 1
        if line.rstrip() in categoryDict:
            continue
        categoryDict[line.rstrip()] = categoryCnt
        categoryCnt = categoryCnt + 1
        
    
    
    
if firstPage == 1:
    if len(sectionText.rstrip()) == 0:
        noOfSecPerPage = noOfSecPerPage - 1
    pageSecCntDict[pageName] = noOfSecPerPage



# In[11]:

#vectorizer = CountVectorizer(stop_words =  stopwords_filtered, max_features = 100000, ngram_range = (1,3))
print "Training TF-IDF vector..."
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words =  stopwords_filtered, max_features = 101938,tokenizer=LemmaTokenizer())
X = vectorizer.fit_transform(corpus)
print "Training TF-IDF vector completed..."


# In[12]:

print "Generating the values of TF_IDF..."
idf = vectorizer.idf_
TfIdfVal = dict(zip(vectorizer.get_feature_names(), idf))
TfIdfSort = sorted(TfIdfVal.items(), key=lambda x: (-x[1], x[0]))

tfIdfDictFile = open("/home/khushboo/wiki10_corpus/preprocessing/TFIDFDictFile.txt", 'w')
for i in TfIdfSort:
    tfIdfDictFile.write(i[0].encode('utf-8') + " : " + str(i[1]) + "\n")
print "Value generation completed..."

tfIdfDictFile.close()

# In[13]:

print "Building word ID dictionary..."

wordCnt = 1
for key in TfIdfSort:
    
    wordDict[key[0]] = wordCnt
    wordCnt = wordCnt + 1

print "Vocab Size: " + str(wordCnt)
vocabSizeFile.write("Vocab Size: " + str(wordCnt)+"\n")
print "Building word ID dictionary completed..."


# In[14]:

typeOfLineReadFromFile = 0
sectionContd = 0
firstPage = 1
#print pageCnt
outputFile.write(str(pageCnt)+'\n')
sectionData = ""

# In[15]:

print "Staring the final file parsing..."

inputFile = open(inputFileName)
ite = 0
donePrint = 1

for line in inputFile:
    
    #print typeOfLineReadFromFile
    if ite % 50 == 0 and donePrint == 1:
        print "Final parsing: " + str(ite) + " pages parsed..."
    donePrint = 0
    
    if line == "####Page Name####\n":
        ite = ite + 1
        donePrint = 1
        if firstPage == 0:
            #outputFile.write("\n\nThis section contains:\n" + sectionData)
            getWordID(sectionData, wordDict, outputFile)
            
        firstPage = 0
        typeOfLineReadFromFile = 1
        continue
    elif line == "######Section Name######\n":
        if typeOfLineReadFromFile == 3:
            outputFile.write(str(pageSecCntDict[pageName])+"\n")
        else:
            #outputFile.write("\n\nThis section contains:\n" + sectionData)
            getWordID(sectionData, wordDict, outputFile)  
        sectionData = ""
        typeOfLineReadFromFile = 2
        continue
    elif line == "#####Categories#####\n":
        outputFile.write(str(pageCategCntDict[pageName])+"\n")
        typeOfLineReadFromFile = 3
        continue  
    
    if typeOfLineReadFromFile == 0:
        #print line
        
        if len(line.rstrip()) == 0:
            continue
            
        sectionData = sectionData + line
        #print sentenceWords
        #for w in sentenceWords:
            #print w
            #print clean_str(w)
    
    elif typeOfLineReadFromFile == 1:
        pageName = line.rstrip()
        outputFile.write(str(pageDict[line.rstrip()])+"\n")
        #print pageDict[line.rstrip()]
        typeOfLineReadFromFile = 0
        
        
    elif typeOfLineReadFromFile == 2:
        typeOfLineReadFromFile = 0
        continue
        
    elif typeOfLineReadFromFile == 3:
        outputFile.write(line.rstrip()+"\n")
        
getWordID(sectionData, wordDict, outputFile)


# In[16]:


print "Writing dictionary data to files..."
#printing dictionaries
sorted_pageDict = sorted(pageDict.items(), key=operator.itemgetter(1))
#print sorted_pageDict

for i in sorted_pageDict:
    pageMappingFile.write(i[0] + " : " + str(i[1]) + "\n")
    #print i[0] + " : " + str(i[1]) + "\n"
    
sorted_categoryDict = sorted(categoryDict.items(), key=operator.itemgetter(1))

for i in sorted_categoryDict:
    categoryMappingFile.write(i[0] + " : " + str(i[1]) + "\n")


# In[17]:

inputFile.close()
pageMappingFile.close()
categoryMappingFile.close()
outputFile.close()


# In[18]:

print "Finishing preprocessing code..."

