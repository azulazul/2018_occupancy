# Arturo Rodriguez Cristerna
# arturocristerna@gmail.com
# utility functions
import numpy as np
import math
#---------------------------------------------
#definition of normalization function - Training
def minmaxnormtr(myvec):
    mymin=np.min(myvec)
    mymax=np.max(myvec)
    if mymin != mymax:
        norm = (myvec-mymin)/(mymax-mymin)
    else:
        norm = myvec-mymin + 1;
    return {'mynorm':norm, 'mymin':mymin ,'mymax':mymax }
#---------------------------------------------
#definition of normalization function - Testing
def minmaxnormtt(myvec,mymin,mymax):
    if mymin != mymax:
        norm = (myvec-mymin)/(mymax-mymin)
    else:
        norm = myvec-mymin + 1;
    return {'mynorm':norm}
#---------------------------------------------
#definition of error metrics
def berror_metrics(tn,fp,fn,tp):
    tn=tn+0.1 #to overcome errors
    fp=fp+0.1
    fn=fn+0.1
    tp=tp+0.1
    acc = (tn+tp)/(tn+fp+fn+tp);
    err = 1-acc
    sen = tp / (tp+fn);
    spe = tn / (tn+fp);
    auc = (sen + spe) / 2
    mcc = ((tp*tn)+(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return {'acc':acc, 'err':err, 'sen':sen ,'spe':spe ,'auc':auc,'mcc':mcc}
#---------------------------------------------
#definition of Bayes error based on NN - Training
def bayeserrnn(err_nn):
    bayeserr = 0.5*(1.0-math.sqrt(1.0-2.0*err_nn))
    return bayeserr
def bayesaccnn(acc_nn):
    err_nn = 1.0 - acc_nn;
    bayeserr = 0.5*(1.0-math.sqrt(1.0-2.0*err_nn))
    bayesacc = 1-bayeserr;
    return bayesacc
def bayesmccnn(auc_nn):
    err_nn = 1.0 - (auc_nn+1.0)/2.0;
    bayeserr = 0.5*(1.0-math.sqrt(1.0-2.0*err_nn))
    bayesacc = 1-bayeserr;
    bayesauc = bayesacc*2.0 - 1.0;
    return bayesauc
def bayesmccnn_std(auc_nn_std):
    err_nn = (auc_nn_std / 2.0);
    bayeserr = 0.5*(1.0-math.sqrt(1.0-2.0*err_nn))
    bayesmcc_std = bayeserr * 2.0;
    return bayesmcc_std
