# Arturo Rodriguez Cristerna
# Nearest Neighbor and Bayes error with non-parametric method
# all features
import tensorflow as tf
import numpy as np
import math
from datetime import datetime
from sklearn.metrics import confusion_matrix
from myfuncs import *
###############################
print "init of data loading"
train = np.loadtxt("datatraining.txt", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7), dtype=object,
                   converters={1: lambda x: datetime.strptime(x, "\"%Y-%m-%d %H:%M:%S\""),
                               2: np.float,
                               3: np.float,
                               4: np.float,
                               5: np.float,
                               6: np.float,
                               7: np.float})
test1 = np.loadtxt("datatest.txt", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7), dtype=object,
                   converters={1: lambda x: datetime.strptime(x, "\"%Y-%m-%d %H:%M:%S\""),
                               2: np.float,
                               3: np.float,
                               4: np.float,
                               5: np.float,
                               6: np.float,
                               7: np.float})
test2 = np.loadtxt("datatest2.txt", delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7), dtype=object,
                   converters={1: lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
                               2: np.float,
                               3: np.float,
                               4: np.float,
                               5: np.float,
                               6: np.float,
                               7: np.float})
###############################
rows=train.shape[0]
cols=train.shape[1]
xttrain = np.zeros((rows,cols))
yttrain = np.zeros((rows,1))
for i in range(0, rows):
    yttrain[i,0] = train[i,cols-1]
    xttrain[i,0] = float(train[i,0].weekday()>4) #is weekend
    xttrain[i,1] = train[i,0].hour*60*60+train[i,0].minute*60+train[i,0].second #get seconds
    for j in range(1, cols-1):
        xttrain[i,1+j]=train[i,j]
###############################
rows=test1.shape[0]
cols=test1.shape[1]
xttest1 = np.zeros((rows,cols))
yttest1 = np.zeros((rows,1))
for i in range(0, rows):
    yttest1[i,0] = test1[i,cols-1]
    xttest1[i,0]=float(test1[i,0].weekday()>4) #is weekend
    xttest1[i,1]=test1[i,0].hour*60*60+test1[i,0].minute*60+test1[i,0].second #get seconds
    for j in range(1, cols-1):
        xttest1[i,1+j]=test1[i,j]
###############################
rows=test2.shape[0]
cols=test2.shape[1]
xttest2 = np.zeros((rows,cols))
yttest2 = np.zeros((rows,1))
for i in range(0, rows):
    yttest2[i,0] = test2[i,cols-1]
    xttest2[i,0]=float(test2[i,0].weekday()>4) #is weekend
    xttest2[i,1]=test2[i,0].hour*60*60+test2[i,0].minute*60+test2[i,0].second #get seconds
    for j in range(1, cols-1):
        xttest2[i,1+j]=test2[i,j]
###############################
print "end of data loading"
###############################
print "init min-max normalization"
nfeats=7; #number of features;
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
print "init of nearest neighbor"

xtr = tf.placeholder("float", [None, nfeats])#training data in tf
xte = tf.placeholder("float", [nfeats])#testing1 data in tf
# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.argmin(distance, 0)
init = tf.global_variables_initializer()
# vector of predictions
yttest1_pred = np.zeros((yttest1.shape[0],1))
yttest2_pred = np.zeros((yttest2.shape[0],1))
with tf.Session() as sess:
    sess.run(init)
    # Get nearest neighbor for test1
    for i in range(0, xttest1.shape[0]):
        nn_index = sess.run(pred, feed_dict={xtr: xttrain, xte: xttest1[i, :]})
        yttest1_pred[i,0]=yttrain[nn_index,0];
    cnf_matrix1 = confusion_matrix(yttest1, yttest1_pred)
    tn1, fp1, fn1, tp1 = confusion_matrix(yttest1, yttest1_pred).ravel()
    berrors1 = berror_metrics(tn1, fp1, fn1, tp1)
    #print "for test data 1"
    #print "error", berrors1['err'], "sen ", berrors1['sen'], "spe ", berrors1['spe'], "auc ", berrors1['auc'], "mcc1 ", berrors1['mcc']

    # Get nearest neighbor for test2
    for i in range(0, xttest2.shape[0]):
        nn_index = sess.run(pred, feed_dict={xtr: xttrain, xte: xttest2[i, :]})
        yttest2_pred[i,0]=yttrain[nn_index,0];
    cnf_matrix2 = confusion_matrix(yttest2, yttest2_pred)
    tn2, fp2, fn2, tp2 = confusion_matrix(yttest2, yttest2_pred).ravel()
    berrors2 = berror_metrics(tn2, fp2, fn2, tp2)
    #print "for test data 2"
    #print "error", berrors2['err'], "sen ", berrors2['sen'], "spe ", berrors2['spe'], "auc ", berrors2['auc'], "mcc ", berrors2['mcc']

    errors = np.zeros((2,1))
    errors[0,0] = berrors1["err"]
    errors[1,0] = berrors2["err"]
    senss = np.zeros((2,1))
    senss[0,0] = berrors1["sen"]
    senss[1,0] = berrors2["sen"]
    specs = np.zeros((2,1))
    specs[0,0] = berrors1["spe"]
    specs[1,0] = berrors2["spe"]
    aucs = np.zeros((2,1))
    aucs[0,0] = berrors1["auc"]
    aucs[1,0] = berrors2["auc"]
    mccs = np.zeros((2,1))
    mccs[0,0] = berrors1["mcc"]
    mccs[1,0] = berrors2["mcc"]

    print "error: ", errors.mean(), np.std(errors)
    print "sens: ", senss.mean(), np.std(senss)
    print "spec: ", specs.mean(), np.std(specs)
    print "auc: ", aucs.mean(), np.std(aucs)
    print "mcc: ", mccs.mean(), np.std(mccs)

    print "Bayes error", bayeserrnn(errors.mean()), bayeserrnn(np.std(errors))
    print "Bayes sens: ", bayesaccnn(senss.mean()), bayeserrnn(np.std(senss))
    print "Bayes specs: ", bayesaccnn(specs.mean()), bayeserrnn(np.std(specs))
    print "Bayes aucs: ", bayesaccnn(aucs.mean()), bayeserrnn(np.std(aucs))
    print "Bayes mccs: ", bayesmccnn(mccs.mean()), bayesmccnn_std(np.std(mccs))
