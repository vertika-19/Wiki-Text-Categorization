import tensorflow as tf
import math


class OneVsAll:
    def __init__(self,labels,vocabularySize):
        '''
        Constructor
        '''   
        self.vocabularySize=vocabularySize
        self.labels=labels
        
        self.document = tf.placeholder(tf.int32,[None,self.vocabularySize],name="documentPlaceholder")
        self.target = tf.placeholder(tf.float32,[None,self.labels],name="target")
        self.allWeights = tf.Variable(tf.truncated_normal( [self.vocabularySize,self.labels], stddev=0.1),name="allWeights")
        self.bias = tf.Variable(tf.constant(0.1, shape=[1,self.labels]),name="bias")
        self.device = "cpu"
        self.graph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
    def graph(self):
        device_name=self.device
        with tf.device(device_name): 
            self.prediction = linearClassifier()
            print(self.prediction.get_shape().as_list())
            # for i in range(labels):
            #     cost = tf.nn.l2_loss( self.prediction[] - self.target, name="squared_error_cost" )
            #     learningRate = tf.train.exponential_decay(learning_rate=0.0008,
            #                               global_step = 1,
            #                               decay_steps = shape(self.document)[0],
            #                               decay_rate = 0.95,
            #                               staircase = True)
            #     self.optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
    
    def linearClassifier(self):
        return tf.sigmoid(tf.matmul(self.document, self.allWeights) + self.bias )
    
    def train(self,data):
        feed_dict_input={}
        feed_dict_input[self.target]=data[0]
        feed_dict_input[self.document]= data[1]
        _, cost = self.session.run((self.optimizer,self.cost),feed_dict=feed_dict_input)
        return cost

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



