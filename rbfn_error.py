# -*- coding: utf-8 -*-
# Arturo Rodriguez Cristerna
# arturocristerna@gmail.com
import numpy as np
import math
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from myfuncs import *
from rbfn import RBFN

###############################
print "init of data loading"
train = np.loadtxt("datatraining.txt", delimiter=",", skiprows=1, usecols=(2,3,4,5,6,7), dtype=object,
                   converters={2: np.float,
                               3: np.float,
                               4: np.float,
                               5: np.float,
                               6: np.float,
                               7: np.int})
test1 = np.loadtxt("datatest.txt", delimiter=",", skiprows=1, usecols=(2,3,4,5,6,7), dtype=object,
                   converters={2: np.float,
                               3: np.float,
                               4: np.float,
                               5: np.float,
                               6: np.float,
                               7: np.int})
test2 = np.loadtxt("datatest2.txt", delimiter=",", skiprows=1, usecols=(2,3,4,5,6,7), dtype=object,
                   converters={2: np.float,
                               3: np.float,
                               4: np.float,
                               5: np.float,
                               6: np.float,
                               7: np.int})
###############################
rows=train.shape[0]
cols=train.shape[1]
xttrain = np.zeros((rows,cols-1))
yttrain = np.zeros((rows,1)).astype(int)
for i in range(0, rows):
    yttrain[i,0] = train[i,cols-1]
    for j in range(0, cols-1):
        xttrain[i,j]=train[i,j]
###############################
rows=test1.shape[0]
cols=test1.shape[1]
xttest1 = np.zeros((rows,cols-1))
yttest1 = np.zeros((rows,1)).astype(int)
for i in range(0, rows):
    yttest1[i,0] = test1[i,cols-1]
    for j in range(0, cols-1):
        xttest1[i,j]=test1[i,j]
###############################
rows=test2.shape[0]
cols=test2.shape[1]
xttest2 = np.zeros((rows,cols-1))
yttest2 = np.zeros((rows,1)).astype(int)
for i in range(0, rows):
    yttest2[i,0] = test2[i,cols-1]
    for j in range(0, cols-1):
        xttest2[i,j]=test2[i,j]
###############################
print "end of data loading"
###############################
print "init min-max normalization"
nfeats=5; #number of features;
#training the normalization step
xttrain_min=np.zeros((nfeats,1))
xttrain_max=np.zeros((nfeats,1))
for i in range(0, nfeats):
    mytmp=minmaxnormtr(xttrain[:,i])
    xttrain[:,i] = mytmp['mynorm']
    xttrain_min[i,0] = mytmp['mymin']
    xttrain_max[i,0] = mytmp['mymax']
#transforming the data with the normalization values
for i in range(0, nfeats):
    mytmp = minmaxnormtt(xttest1[:,i], xttrain_min[i,0], xttrain_max[i,0])
    xttest1[:,i] = mytmp['mynorm']
    mytmp = minmaxnormtt(xttest2[:,i], xttrain_min[i,0], xttrain_max[i,0])
    xttest2[:,i] = mytmp['mynorm']
pos_class = yttrain == 1;
neg_class = yttrain == 0;
pos_class_tot = pos_class.sum();
neg_class_tot = neg_class.sum();
imbalance_ratio1 = float(pos_class_tot) / float(neg_class_tot);
imbalance_ratio2 = float(neg_class_tot) / float(pos_class_tot);
print "positive class: ", pos_class_tot, " | negative class: ", neg_class_tot, " | imbalance ratio1: ", imbalance_ratio1, " | imbalance ratio2: ", imbalance_ratio2
print "end min-max normalization"
###############################
myrbfn_path="./rbfn_model.dat";
#for training and saving the model
#np.ceil(np.sqrt(xttest1.shape[0])).astype(int)
#model = RBFN(hidden_n = 10, output_n = 2, t_sigmas=0, n_centers=2)
#model.train(xttrain,yttrain,myrbfn_path);
#for loading the prevously trained model
model = joblib.load(myrbfn_path)

yttest1_pred  = model.predict(xttest1)
cnf_matrix1 = confusion_matrix(yttest1[:,0], yttest1_pred['class'][:,0])
tn1, fp1, fn1, tp1 = cnf_matrix1.ravel()
berrors1 = berror_metrics(tn1, fp1, fn1, tp1)
#print "for test data 1"
#print "acc", berrors1['acc'], "sen ", berrors1['sen'], "spe ", berrors1['spe'], "auc ", berrors1['auc'], "mcc ", berrors1['mcc']

yttest2_pred = model.predict(xttest2)
cnf_matrix2 = confusion_matrix(yttest2, yttest2_pred['class'])
tn2, fp2, fn2, tp2 = cnf_matrix2.ravel()
berrors2 = berror_metrics(tn2, fp2, fn2, tp2)
#print "for test data 2"
#print "acc", berrors2['acc'], "sen ", berrors2['sen'], "spe ", berrors2['spe'], "auc ", berrors2['auc'], "mcc ", berrors2['mcc']

accs_ = np.zeros((2,1))
accs_[0,0] = berrors1["acc"]
accs_[1,0] = berrors2["acc"]
errors_ = np.zeros((2,1))
errors_[0,0] = berrors1["err"]
errors_[1,0] = berrors2["err"]
senss_ = np.zeros((2,1))
senss_[0,0] = berrors1["sen"]
senss_[1,0] = berrors2["sen"]
specs_ = np.zeros((2,1))
specs_[0,0] = berrors1["spe"]
specs_[1,0] = berrors2["spe"]
aucs_ = np.zeros((2,1))
aucs_[0,0] = berrors1["auc"]
aucs_[1,0] = berrors2["auc"]
mccs_ = np.zeros((2,1))
mccs_[0,0] = berrors1["mcc"]
mccs_[1,0] = berrors2["mcc"]
print "Classifier metric: mean ± standard deviation"
print "SVM error: ", accs_.mean(), "±", np.std(accs_)
print "SVM sens: ", senss_.mean(), "±", np.std(senss_)
print "SVM spec: ", specs_.mean(), "±", np.std(specs_)
print "SVM auc: ", aucs_.mean(), "±", np.std(aucs_)
print "SVM mcc: ", mccs_.mean(), "±", np.std(mccs_)
