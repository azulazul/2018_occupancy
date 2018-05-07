# Arturo Rodriguez
# arturocristerna@gmail.com
# Training SVM with rbf kernel, balanced weights and grid search

import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.externals import joblib

class RBFN(object):

    def __init__(self, hidden_n,output_n,t_sigmas=0,n_centers=2):
        """ radial basis function network with k-means and SVD
        #
            hidden_n: number of neurons in hidden layer
            output_n: number of neurons of output layer; number of classes
            sigmas: 0, geometric sigma; 1, uniform sigma
            p_centers:  number of centers used to compute the sigmas; default value 2
        """
        self.hidden_n = hidden_n
        self.output_n = output_n
        self.kmeans = None
        self.sigmas = None
        self.weights = None
        self.t_sigmas = t_sigmas
        self.n_centers = n_centers
    def train(self,xttrain,yttrain,model_output_path):
        #xttrain matrix(number of samples, number of atributes)
        #yttrain is a vector (number of samples,1)
        self.kmeans = KMeans(n_clusters=self.hidden_n, init='random').fit(xttrain)
        #print self.kmeans.cluster_centers_
        if self.t_sigmas==0:
            self.sigmas = self._geometricSigma_(self.kmeans.cluster_centers_)
        else:
            self.sigmas = self._uniformSigma_(self.kmeans.cluster_centers_,self.n_centers)
        #print self.sigmas
        theta=self._activationHidden_(self.kmeans.cluster_centers_,self.sigmas,xttrain,xttrain.shape[0])
        #print theta
        yttrain = yttrain.astype(int)
        trOutputLayer=self._yOuputLayer_(yttrain,self.output_n)
        #print trOutputLayer
        theta_mod=np.matmul(np.linalg.pinv(np.matmul(theta,np.transpose(theta))),theta)
        self.weights = np.matmul(theta_mod,trOutputLayer)
        #print self.weights
        if len(model_output_path)>0:
            joblib.dump(self, model_output_path)
    def _uniformSigma_(self,cluster_centers,p_centers=2):
        nclusters =cluster_centers.shape[0]
        pdist = np.zeros((nclusters,nclusters))
        s = np.zeros((nclusters,1))
        for i in range(0,nclusters-1):
            pdist[i,i] = np.inf
            for j in range(i+1,nclusters):
                pdist[j,j] = np.inf
                pdist[i,j] = euclidean(cluster_centers[i,:],cluster_centers[j,:])
                pdist[j,i] = pdist[i,j]
        pdist2=np.sort(pdist, axis=0)
        for i in range(0,nclusters):
            s[i,0]=np.mean(pdist2[0:p_centers,i])
        return s

    def _geometricSigma_(self,cluster_centers):
        nclusters = cluster_centers.shape[0]
        pdist = np.zeros((nclusters,nclusters))
        s = np.zeros((nclusters,1))
        for i in range(0,nclusters-1):
            pdist[i,i] = np.inf
            for j in range(i+1,nclusters):
                pdist[j,j] = np.inf
                pdist[i,j] = euclidean(cluster_centers[i,:],cluster_centers[j,:])
                pdist[j,i] = pdist[i,j]
        pdist2=np.sort(pdist, axis=0)
        for i in range(0,nclusters):
            s[i,0]=np.sqrt(pdist2[0,i]*pdist2[1,i])
        return s

    def _activationHidden_(self,cluster_centers,sigmas,trData,ninstances):
        #Gaussian activation kernel
        nclusters = cluster_centers.shape[0]
        sigmas_mod = -2*np.square(sigmas)
        theta=np.ones((nclusters+1,ninstances))
        for i in range(1,nclusters+1):
            for j in range(0,ninstances):
                if ninstances==1:
                    theta[i,j] = np.exp(euclidean(trData[:],cluster_centers[i-1,:])/sigmas_mod[i-1,0])
                else:
                    theta[i,j] = np.exp(euclidean(trData[j,:],cluster_centers[i-1,:])/sigmas_mod[i-1,0])
        return theta

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

    def predict(self,mytest):
        """
        # output
        'class' - predicted class
        'probs' - probabilites for each class and instance
        'myprob' - probabilites for each instance
        """

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
        mytest_hr=self._activationHidden_(self.kmeans.cluster_centers_,self.sigmas,mytest,nsamples)
        mytest_or = np.matmul(np.transpose(self.weights),mytest_hr)
        mytest_or = np.transpose(mytest_or)
        mytest_tot = np.sum(mytest_or,myaxis)
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
