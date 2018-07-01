import tensorflow as tf
import numpy as np
import math

TotalNoOfInstances = 107016

class clustering_DL:
    def __init__(self,maxParagraphs,paragraphLength,labels,vocabularySize,filterSizes,num_filters\
        ,wordEmbeddingDimension,lrate,poolLength, keep_prob):
        self.wordEmbeddingDimension = wordEmbeddingDimension
        self.vocabularySize=vocabularySize
        self.labels=labels
        self.filterSizes = filterSizes
        self.num_filters = num_filters
        self.paragraphLength = paragraphLength
        self.maxParagraph = maxParagraphs
        self.poolLength = poolLength
        self.numberOfClusters = 32
        self.noOfLabelsPerClusters = open("cluster_metainfo_wiki10.txt").read().strip().split("\n")
        self.noOfLabelsPerClusters = [int(x) for x in self.noOfLabelsPerClusters]
        self.clusterLayerInput = int(len(self.filterSizes)*self.num_filters*( \
                                                math.ceil(float(self.maxParagraph)/self.poolLength)  ) )
        self.learning_rate = lrate


        self.noOfInstancePerClusters = open("trn_cluster_noofInstances.txt").read().strip().split("\n")
        self.noOfInstancePerClusters = [ TotalNoOfInstances / float(x) for x in self.noOfInstancePerClusters]

        self.weightsForCluster = []
        for x in range(self.numberOfClusters):
            temp = [ self.noOfInstancePerClusters[x] ] * self.noOfLabelsPerClusters[x]
            self.weightsForCluster.extend(temp)


        #added on 21 june 2018 keep prob for dropout- for testing already passing 1.0
        self.keep_prob = keep_prob

        #added on 21june
        tf.set_random_seed(42)

        
        self.device ='gpu'
        self.wordEmbedding = tf.Variable(tf.random_uniform([self.vocabularySize, self.wordEmbeddingDimension], -1.0, 1.0),name="wordEmbedding")

        self.feaIDList = []
        self.feaValueList = []
        for i in range(self.maxParagraph):
            self.feaIDList.append(tf.placeholder(tf.int32,[None,self.paragraphLength],name="featureIDPlaceholder"+str(i)))
            self.feaValueList.append(tf.placeholder(tf.float32,[None,self.paragraphLength],name="featureValuePlaceholder"+str(i)))

        self.target = tf.placeholder(tf.float32,[None,self.labels],name="target")
        self.graph()

        gpu_options = tf.GPUOptions(visible_device_list="1,2")
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session.run(tf.global_variables_initializer())
        
    
    def graph(self):
        device_name=self.device
        with tf.device(device_name): 
            self.convOutput = self.convLayer()
            self.clusterLayerOutput = self.clusterLayer()
            self.prediction = self.outputLayer()
            #print("Size of prediction by model")
            #print(self.prediction.get_shape().as_list())
            self.weightsForCluster = tf.multiply(self.prediction, self.weightsForCluster) 
            #print("Size of weight of final prediction")
            #print(self.weightsForCluster.get_shape().as_list())
            

            self.cross_entropy = -tf.reduce_sum(((self.target*tf.log(self.weightsForCluster + 1e-9)) + ((1-self.target) * tf.log(1 - self.weightsForCluster + 1e-9)) )  , name='xentropy' ) 
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
        # print(document.get_shape().as_list())
        pooled_outputs=[]
        
        filtered_outputs = []
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

            h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            pool_length = self.poolLength
            pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, pool_length, 1, 1],
                        strides=[1, pool_length, 1, 1],
                        padding='SAME',
                        name="pool")
            filtered_outputs.append(pooled)

        filtered_outputs = tf.concat(filtered_outputs,axis = 1 )	#concatenating for different filter sizes - as this will be for 1 doc 
        
        return tf.reshape(filtered_outputs,[-1,self.clusterLayerInput])
    
    def clusterLayer(self):
        #adding dropout
        layer_dropout = tf.nn.dropout( self.convOutput , self.keep_prob )
        shape = [self.clusterLayerInput,self.numberOfClusters]
        weights =tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="CL_W")
        bias = tf.Variable(tf.constant(0.1, shape=[self.numberOfClusters]),name="CL_Bias")
        out_layer = tf.nn.sigmoid(tf.matmul(layer_dropout, weights) + bias)
        return out_layer
    
    def outputLayer(self):

    	final_out_layer = []

    	transposedclusterLayerOutput = tf.transpose(self.clusterLayerOutput)
    	for i in range(self.numberOfClusters):
	        shape = [1,self.noOfLabelsPerClusters[i] ]
	        weights =tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="OUT_W")
	        bias = tf.Variable(tf.constant(0.1, shape=[ self.noOfLabelsPerClusters[i] ] ),name="OUT_Bias")
	        # print(tf.expand_dims(tf.transpose(transposedclusterLayerOutput[i]), -1).get_shape().as_list())
        	# print(weights.get_shape().as_list())
	        out_layer = tf.nn.sigmoid(tf.matmul( tf.expand_dims(tf.transpose(transposedclusterLayerOutput[i]), -1) , weights) + bias)
            # print(out_layer.get_shape().as_list())
	        if i == 0:
	        	final_out_layer = out_layer
	        	continue
	        final_out_layer = tf.concat([final_out_layer, out_layer] ,1)
	        # print(final_out_layer.get_shape().as_list())
        print(final_out_layer.get_shape().as_list())
        return final_out_layer


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
