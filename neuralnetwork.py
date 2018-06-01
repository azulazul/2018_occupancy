import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
class neuralnetwork(object):

    def __init__(self, hidden1_n, hidden2_n, output_n, epoch, interval):
        """ Neural network with two hidden layers and GradientDescentOptimizer
        #
            hidden1_n: number of neurons in the first hidden layer
            hidden2_n: number of neurons in the second hidden layer
            output_n: number of neurons of output layer; number of classes
            epoch: number of epochs
            interval: intervals to get an output
        """
        self.hidden1_n = hidden1_n
        self.hidden2_n = hidden2_n
        self.output_n = output_n
        self.epoch = epoch
        self.interval = interval

    def train(self,xttrain,yttrain,model_output_path,model_output_pathTF):
        seed = 1234
        np.random.seed(seed)
        tf.set_random_seed(seed)
        tf.reset_default_graph()
        sess = tf.Session()

        nfeatures = xttrain.shape[1] # N
        self.nfeatures = nfeatures
        nclasses = self.output_n  # m

        yttrain = yttrain.astype(int)
        y_train =self._yOuputLayer_(yttrain,self.output_n)

        X_data = tf.placeholder(shape=[None, nfeatures], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, nclasses], dtype=tf.float32)
        #print "hiddenLayer1 ", self.hidden1_n
        #print "hiddenLayer2 ", self.hidden2_n
        # Create variables for Neural Network layers
        w1 = tf.Variable(tf.random_normal(shape=[nfeatures,self.hidden1_n ])) # Inputs -> Hidden Layer
        b1 = tf.Variable(tf.random_normal(shape=[self.hidden1_n ]))   # First Bias
        w2 = tf.Variable(tf.random_normal(shape=[self.hidden1_n ,self.hidden2_n])) # Hidden layer 1 -> Hidden layer 2
        b2 = tf.Variable(tf.random_normal(shape=[self.hidden2_n]))   # Second Bias
        wout = tf.Variable(tf.random_normal(shape=[self.hidden2_n,nclasses])) # Hidden layer 2 -> Outputs
        bout = tf.Variable(tf.random_normal(shape=[nclasses]))   # Tirdth Bias
        # Operations
        #relu layers
        #hidden1_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
        #hidden2_output = tf.nn.relu(tf.add(tf.matmul(hidden1_output, w2), b2))
        #sigmoid layers
        hidden1_output = tf.nn.sigmoid(tf.add(tf.matmul(X_data, w1), b1))
        hidden2_output = tf.nn.sigmoid(tf.add(tf.matmul(hidden1_output, w2), b2))
        final_output = tf.nn.softmax(tf.add(tf.matmul(hidden2_output, wout), bout))
        # Cost Function
        loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))
        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

        # Initialize variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)

        # Training
        print('Training the model...')
        for i in range(1, (self.epoch + 1)):
            sess.run(optimizer, feed_dict={X_data: xttrain, y_target: y_train})
            if i % self.interval == 0:
                print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: xttrain, y_target: y_train}))

        if len(model_output_path)>0:
            joblib.dump(self, model_output_path)
        save_path = saver.save(sess, model_output_pathTF)
        #print("Model saved in path: %s" % save_path)
        sess.close()

    def _yOuputLayer_(self,trY,nclasses):
        minc=np.min(trY)
        maxc=np.max(trY)
        if minc<0 or maxc<(nclasses-1):
            raise ValueError('labels out of bounds, they must be in range [0,C)')
        ninstances = trY.shape[0]
        yOuputLayer = np.zeros((ninstances,nclasses))
        for i in range(0,ninstances):
            yOuputLayer[i,trY[i,0]]=1
        return yOuputLayer

    def predict(self,mytest,mymodelTF_path):
        """
        mymodelTF_path - tensorflow session
        # output
        'class' - predicted class
        'probs' - probabilites for each class and instance
        'myprob' - probabilites for each instance
        """
        # Initialize variables
        seed = 1234
        nfeatures = self.nfeatures
        nclasses   = self.output_n
        tf.set_random_seed(seed)
        tf.reset_default_graph()
        X_data = tf.placeholder(shape=[None, nfeatures], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, nclasses], dtype=tf.float32)
        #print "hiddenLayer1 ", self.hidden1_n
        #print "hiddenLayer2 ", self.hidden2_n
        # Create variables for Neural Network layers
        w1 = tf.Variable(tf.random_normal(shape=[nfeatures,self.hidden1_n ])) # Inputs -> Hidden Layer
        b1 = tf.Variable(tf.random_normal(shape=[self.hidden1_n ]))   # First Bias
        w2 = tf.Variable(tf.random_normal(shape=[self.hidden1_n ,self.hidden2_n])) # Hidden layer 1 -> Hidden layer 2
        b2 = tf.Variable(tf.random_normal(shape=[self.hidden2_n]))   # Second Bias
        wout = tf.Variable(tf.random_normal(shape=[self.hidden2_n,nclasses])) # Hidden layer 2 -> Outputs
        bout = tf.Variable(tf.random_normal(shape=[nclasses]))   # Tirdth Bias
        # Operations
        #relu layers
        #hidden1_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
        #hidden2_output = tf.nn.relu(tf.add(tf.matmul(hidden1_output, w2), b2))
        #sigmoid layers
        hidden1_output = tf.nn.sigmoid(tf.add(tf.matmul(X_data, w1), b1))
        hidden2_output = tf.nn.sigmoid(tf.add(tf.matmul(hidden1_output, w2), b2))
        final_output = tf.nn.softmax(tf.add(tf.matmul(hidden2_output, wout), bout))
        # Cost Function
        loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))
        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, mymodelTF_path)
        #print("Model restored.")

        try:
            nsamples   = mytest.shape[0]
            natributes = mytest.shape[1]
            myaxis     = 1
        except:
            nsamples   = 1
            natributes = mytest.shape[0]
            myaxis     = 1

        #print "samples: ", nsamples
        #print "axis: ", myaxis
        #print "classes: ", self.output_n
        mytest_or = sess.run(final_output, feed_dict={X_data: mytest})
        sess.close()

        #print "sums", mytest_tot
        #print "scores", mytest_or
        mytest_orp  = np.zeros((nsamples,self.output_n))
        my_class    = np.zeros((nsamples,1)).astype(int)
        my_prob     = np.zeros((nsamples,1))
        for i in range (0,nsamples):
            #mytest_orp[i,:] = mytest_or[i,:] / mytest_tot[i]
            mytest_orp[i,:] = mytest_or[i,:]
            #print mytest_orp[i,:]
            my_class[i,0]     = np.argmax(mytest_orp[i,:]).astype(int)
            #print my_class[i,0]
            my_prob[i,0]      = mytest_orp[i,my_class[i,0]]
            #print my_prob[i,0]

        return {'class': my_class, 'probs': mytest_orp, 'myprob': my_prob}
