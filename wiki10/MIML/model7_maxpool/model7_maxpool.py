import tensorflow as tf
import numpy as np
import math

class Model7_maxpool:
    def __init__(self,maxParagraphs,paragraphLength,labels,vocabularySize,filterSizes\
    				,num_filters,wordEmbeddingDimension,lrate, keep_prob):
        self.wordEmbeddingDimension = wordEmbeddingDimension
        self.vocabularySize=vocabularySize
        self.labels=labels
        self.filterSizes = filterSizes
        self.num_filters = num_filters
        self.paragraphLength = paragraphLength
        self.maxParagraph = maxParagraphs
        self.fullyConnectedLayerInput = int(self.num_filters)
        self.learning_rate = lrate

        self.keep_prob = keep_prob

        #added on 18june
        tf.set_random_seed(42)
        
        self.device ='cpu'
        self.wordEmbedding = tf.Variable(tf.random_uniform([self.vocabularySize, self.wordEmbeddingDimension], -1.0, 1.0),name="wordEmbedding")

        self.feaIDList = []
        self.feaValueList = []
        for i in range(self.maxParagraph):
            self.feaIDList.append(tf.placeholder(tf.int32,[None,self.paragraphLength],name="featureIDPlaceholder"+str(i)))
            self.feaValueList.append(tf.placeholder(tf.float32,[None,self.paragraphLength],name="featureValuePlaceholder"+str(i)))

        self.target = tf.placeholder(tf.float32,[None,self.labels],name="target")
        self.graph()        
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
    
    def graph(self):
        device_name=self.device
        with tf.device(device_name): 
            self.convOutput = self.convLayer()
            self.prediction = self.fullyConnectedLayer()
            self.cross_entropy = -tf.reduce_sum(((self.target*tf.log(self.prediction + 1e-9)) + ((1-self.target) * tf.log(1 - self.prediction + 1e-9)) )  , name='xentropy' ) 
            self.cost = tf.reduce_mean(self.cross_entropy)
            # self.cost = tf.nn.l2_loss( self.target - self.prediction  )
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    
    def getSectionEmbedding(self,feaIDs,feaValues):
        device_name = self.device
        with tf.device(device_name): 
            secEmbedding = tf.nn.embedding_lookup( self.wordEmbedding,feaIDs )
            weightedFea = tf.tile( tf.expand_dims(feaValues,-1),[1,1,self.wordEmbeddingDimension],name = "wordWeight_tile")            
            secEmbedding = tf.reduce_sum( tf.multiply(secEmbedding,weightedFea), 1, keep_dims=True )
            totalFreqPerPara = tf.reduce_sum(feaValues, 1, keep_dims=True, name = "totalFreqPerPara_reduce_sum")
            secEmbeddingWt = tf.reciprocal( totalFreqPerPara, name = "secEmbedding_reciprocal" )
            secEmbeddingWt = tf.tile( tf.expand_dims( secEmbeddingWt ,-1),[1,1,self.wordEmbeddingDimension],name = "secEmbeddingWt_tile")
            secEmbedding = tf.multiply(secEmbedding,secEmbeddingWt, name = "weighted_avg_multiply")
        return secEmbedding
    
    def convLayer(self):
    
        docEmbedding=[]

        for feaId,feaVal in zip(self.feaIDList,self.feaValueList):
            secEmbedding = self.getSectionEmbedding(feaId,feaVal)
            docEmbedding.append(secEmbedding)
        document = tf.concat(docEmbedding,axis=1)
        document = tf.expand_dims( document , -1)
        # print(document.get_shape().as_list())		[ batch size, maxParagraph( becuase of same padding), word embed dim, 1 ]
        pooled_outputs=[]
        
        filtered_ouputs = []
        for filter_size in self.filterSizes:
            shape = [filter_size,self.wordEmbeddingDimension,1,self.num_filters]
            weights = tf.Variable( tf.truncated_normal( shape, stddev=0.1 ),name="convLayerW_"+str(filter_size))
            bias= tf.Variable( tf.constant( 0.1, shape=[self.num_filters]),name="convLayerB_"+str(filter_size))
            conv = tf.nn.conv2d(
                        document,
                        weights,
                        strides=[1, 1, self.wordEmbeddingDimension, 1],
                        padding="SAME",
                        name="conv")
            # [ batch size, maxParagraph, 1 ( as conv on a full sec embed ) , no of filters ]

            h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            h = tf.nn.max_pool( h, 
                        ksize=[1,self.maxParagraph,1,1],
                        strides=[1,self.maxParagraph,1, 1],
                        padding="SAME",
                        name="maxpool" )

            filtered_ouputs.append(h)
            # [ batch size, 1, 1  , no of filters ]

        filtered_ouputs = tf.concat(filtered_ouputs,axis = 1 )	#concatenating for different filter sizes - as this will be for 1 doc 
        
        return tf.reshape(filtered_ouputs,[-1,self.fullyConnectedLayerInput])
    
    def fullyConnectedLayer(self):
    	# layer = tf.nn.sigmoid(tf.matmul(self.convOutput, weights) + bias)
        
        #adding dropout
        layer_dropout = tf.nn.dropout( self.convOutput , self.keep_prob )

        shape = [self.fullyConnectedLayerInput,self.labels]
        weights =tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="FC_W")
        bias = tf.Variable(tf.constant(0.1, shape=[self.labels]),name="FC_Bias")

        out_layer = tf.nn.sigmoid(tf.matmul(layer_dropout, weights) + bias)
        return out_layer
    
    def train(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.feaIDList[p]] = data[1][p]
            feed_dict_input[self.feaValueList[p]]= data[2][p]

        _, cost = self.session.run((self.optimizer,self.cost),feed_dict=feed_dict_input)

        return cost

    def predict(self,data):
        feed_dict_input={}
        for p in range(self.maxParagraph):
            feed_dict_input[self.feaIDList[p]]= data[1][p]
            feed_dict_input[self.feaValueList[p]]= data[2][p]
            
        pred=self.session.run(self.prediction,feed_dict=feed_dict_input)
        return pred
    
    def getError(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.feaIDList[p]]= data[1][p]
            feed_dict_input[self.feaValueList[p]]= data[2][p]
        cost = self.session.run(self.cost,feed_dict=feed_dict_input)
        return cost 

    def save(self,save_path):
        saver = tf.train.Saver()
        saver.save(self.session, save_path)


    def load(self,save_path):
        self.session = tf.Session()
        new_saver = tf.train.Saver()
        new_saver.restore(self.session, save_path)


    def save_label_embeddings(self):
        pass
