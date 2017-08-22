import tensorflow as tf
import math

class Model7:
    def __init__(self,maxParagraphs,paragraphLength,labels,vocabularySize):
        '''
        Constructor
        '''
        self.wordEmbeddingDimension = 50
        self.vocabularySize=vocabularySize
        self.labels=labels
        self.filterSizes_doc = [2,3]
        self.num_filters_doc=25
        self.paragraphLength = paragraphLength
        self.maxParagraph = maxParagraphs
        self.poolLength=10
        self.fullyConnectedLayerInput = int(len(self.filterSizes_doc)*self.num_filters_doc)
        
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
            self.convOutput=self.convLayer()
            self.prediction=self.fullyConnectedLayer(self.convOutput,self.labels)
            self.cross_entropy = -tf.reduce_sum(((self.target*tf.log(self.prediction + 1e-9)) + ((1-self.target) * tf.log(1 - self.prediction + 1e-9)) )  , name='xentropy' ) 
            self.cost = tf.reduce_mean(self.cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        
    
    def getSectionEmbedding(self,feaIDs,feaValues):
        device_name = self.device
        with tf.device(device_name): 
            secEmbedding = tf.nn.embedding_lookup( self.wordEmbedding,feaIDs )
            weightedFea =tf.tile(tf.expand_dims(feaValues,-1),[1,1,self.wordEmbeddingDimension],name = "wordWeight")
            secEmbedding = tf.multiply(secEmbedding,weightedFea)
            # print( secEmbedding.get_shape().as_list() )
        return tf.expand_dims(secEmbedding,-1)
    
    
    
    # def convLayeronParagraph(self,paragraphVector,filterSizes,num_input_channels,num_filters):
    
    #     pooled_outputs=[]
    #     for filter_size in filterSizes:
    #         shape = [filter_size,self.wordEmbeddingDimension,1,num_filters]

    #         weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="paragraphConvLayerW_"+str(filter_size))
    #         bias= tf.Variable(tf.constant(0.1, shape=[num_filters]),name="paragraphConvLayerB_"+str(filter_size))
    #         conv = tf.nn.conv2d(
    #                     paragraphVector,
    #                     weights,
    #                     strides=[1, 1, self.wordEmbeddingDimension, 1],
    #                     padding="SAME",
    #                     name="conv")

    #         h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
    #         pool_length=self.poolLength
    #         pooled = tf.nn.max_pool(
    #                     h,
    #                     ksize=[1, pool_length, 1, 1],
    #                     strides=[1, pool_length, 1, 1],
    #                     padding='SAME',
    #                     name="pool")
    #         pooled_outputs.append(pooled)
    #     return tf.concat(pooled_outputs,axis=1)

    
    
    def convLayer(self):
    
        docEmbedding=[]

        for feaId,feaVal in zip(self.feaIDList,self.feaValueList):
            secEmbedding = self.getSectionEmbedding(feaId,feaVal)
            docEmbedding.append(secEmbedding)
        print(docEmbedding[0])

        doc = tf.concat(docEmbedding,axis=1)
        #doc = tf.reshape(doc,[-1,self.convLayerOutputSize,self.num_filters_doc,1])
        # pooled_outputs=[]
        '''
        for filter_size in filterSizes_doc:
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
        '''

        # shape = self.filterShapeOfAllPara

        # weights= tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="paragraphConvLayer2W_"+str(filterShapeOfAllPara[0]))
        # bias= tf.Variable(tf.constant(0.1, shape=[num_filters_allPara]),name="paragraphConvLayer2B_"+str(filterShapeOfAllPara[0]))

        # conv = tf.nn.conv2d(
        #                 self.allParagraph,
        #                 weights,
        #                 strides=[1, 1, 1, 1],
        #                 padding="SAME",
        #                 name="conv")
        # h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
        return tf.reshape(docEmbedding,[-1,self.fullyConnectedLayerInput])
    
    def fullyConnectedLayer(self,convOutput,labels):
        shape = [self.fullyConnectedLayerInput,labels]
        weights =tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="FC_W")
        bias = tf.Variable(tf.constant(0.1, shape=[labels]),name="FC_Bias")
        layer = tf.nn.sigmoid(tf.matmul(convOutput, weights) + bias)
        return layer
    
    def train(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        for p in range(self.maxParagraph):
            feed_dict_input[self.feaIDList[p]]= data[1][p]
            feed_dict_input[self.feaValueList[p]]= data[2][p]

        _, cost = self.session.run((self.optimizer,self.cost),feed_dict=feed_dict_input)
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



# In[26]:

#maxParagraphLength=35
#maxParagraphs=5
#labels=10
#vocabularySize=150

#model = Model(maxParagraphLength,maxParagraphs,labels,vocabularySize)


# In[27]:

#model.graph()


# In[ ]:


