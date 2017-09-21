import tensorflow as tf
import math


class Model3:
    def __init__(self,maxParagraphs,paragraphLength,labels,vocabularySize,filterSizes,num_filters,poolLength,wordEmbeddingDimension,lrate):
        self.wordEmbeddingDimension = wordEmbeddingDimension   
        self.vocabularySize=vocabularySize
        self.labels=labels
        self.filterSizes_paragraph = filterSizes
        self.paragraphLength= paragraphLength
        self.num_filters_parargaph= num_filters
        self.maxParagraph = maxParagraphs
        self.poolLength= poolLength

        self.paragraphOutputSize = len(self.filterSizes_paragraph)*self.num_filters_parargaph*int(math.ceil(paragraphLength/float(self.poolLength)))
        self.conv2LayerOutputSize = len(self.filterSizes_paragraph)*int(math.ceil(paragraphLength/float(self.poolLength)))*maxParagraphs
        self.fullyConnectedLayerInput = self.paragraphOutputSize
        self.wordEmbedding = tf.Variable(tf.random_uniform([self.vocabularySize, self.wordEmbeddingDimension], -1.0, 1.0),name="wordEmbedding")
        self.learning_rate = lrate


        self.paragraphList = []
        for i in range(self.maxParagraph):
            self.paragraphList.append(tf.placeholder(tf.int32,[None,self.paragraphLength],name="paragraphPlaceholder"+str(i)))

        self.target = tf.placeholder(tf.float32,[None,self.labels],name="target")
        
        self.device = "cpu"
        self.graph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
    
    def graph(self):
        device_name=self.device
        with tf.device(device_name): 
            self.convOutput=self.convLayerCombineParagraph()
            self.prediction=self.fullyConnectedLayer()
            self.cross_entropy = -tf.reduce_sum(((self.target*tf.log(self.prediction + 1e-9)) + ((1-self.target) * tf.log(1 - self.prediction + 1e-9)) )  , name='xentropy' ) 
            self.cost = tf.reduce_mean(self.cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    
    def getParagraphEmbedding(self,paragraphWords):
        device_name=self.device
        with tf.device(device_name): 
            paraEmbedding=tf.nn.embedding_lookup(self.wordEmbedding,paragraphWords)
    
        return tf.expand_dims(paraEmbedding, -1)
    
    
    
    def convLayeronParagraph(self,paragraphVector,filterSizes,num_input_channels,num_filters):
    
        pooled_outputs=[]
        for filter_size in filterSizes:
            shape = [filter_size,self.wordEmbeddingDimension,1,num_filters]

            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="paragraphConvLayerW_"+str(filter_size))
            bias= tf.Variable(tf.constant(0.1, shape=[num_filters]),name="paragraphConvLayerB_"+str(filter_size))
            conv = tf.nn.conv2d(
                        paragraphVector,
                        weights,
                        strides=[1, 1, self.wordEmbeddingDimension, 1],
                        padding="SAME",
                        name="conv")

            h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            pool_length=self.poolLength
            pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, pool_length, 1, 1],
                        strides=[1, pool_length, 1, 1],
                        padding='SAME',
                        name="pool")
            pooled_outputs.append(pooled)
        return tf.concat(axis =1,values = pooled_outputs)

    
    
    def convLayerCombineParagraph(self):
    
        paragraphCNNEmbedding=[]

        for paragraph in self.paragraphList:
            paragraphVector = self.getParagraphEmbedding(paragraph)
            cnnEmbedding = self.convLayeronParagraph(paragraphVector,self.filterSizes_paragraph,1,self.num_filters_parargaph)
            paragraphCNNEmbedding.append(tf.reshape(cnnEmbedding,[-1,1,self.paragraphOutputSize]))
            
        allParagraph = tf.reduce_max(tf.concat(paragraphCNNEmbedding,axis=1),axis=1)
        return allParagraph

    def fullyConnectedLayer(self):
        shape = [self.fullyConnectedLayerInput,self.labels]
        weights =tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="FC_W")
        bias = tf.Variable(tf.constant(0.1, shape=[self.labels]),name="FC_Bias")
        layer = tf.nn.sigmoid(tf.matmul(self.convOutput, weights) + bias)
        return layer
    
    def train(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.paragraphList[p]]= data[1][p]
        _, cost = self.session.run((self.optimizer,self.cost),feed_dict=feed_dict_input)

        # tf.summary.scalar('cost', cost)
        return cost

    def predict(self,data):
        feed_dict_input={}
#         feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.paragraphList[p]]= data[1][p]
            
        pred=self.session.run(self.prediction,feed_dict=feed_dict_input)
        return pred
    
    def getError(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.paragraphList[p]]= data[1][p]
        cost = self.session.run(self.cost,feed_dict=feed_dict_input)
        return cost 

    def save(self,save_path):
        saver = tf.train.Saver()
        saver.save(self.session, save_path)


    def load(self,save_path):
        self.session = tf.Session()
#         new_saver = tf.train.import_meta_graph(save_path)
        new_saver = tf.train.Saver()
        new_saver.restore(self.session, save_path)


    def save_label_embeddings(self):
        pass



