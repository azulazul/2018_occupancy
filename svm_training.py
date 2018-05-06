# Arturo Rodriguez
# arturocristerna@gmail.com
# Training SVM with rbf kernel, balanced weights and grid search

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from myfuncs import *
def train_bin(X_train, y_train, model_output_path, deb):
    if deb!=0:
        print "Debugging version"
    base=2
    C_range = np.array([base**-1, base**0, base**1, base**2, base**3, base**4])
    gamma_range = np.array([base**-4, base**-3, base**-2, base**-1, base**0, base**1])
    #C_range = np.array([base**0, base**1])
    #gamma_range = np.array([base**-2, base**-1])

    param_grid = dict(gamma=gamma_range, C=C_range)    
    #k-fold
    cv   = StratifiedKFold(n_splits=5);    
    grid =  GridSearchCV(SVC(class_weight='balanced', kernel='rbf'), param_grid=param_grid, cv=cv, n_jobs=8)
    grid.fit(X_train, y_train.ravel())
    y_train_pred=grid.predict(X_train)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_train, y_train_pred).ravel()
    berrors1 = berror_metrics(tn1, fp1, fn1, tp1)
    if deb!=0:
        print "The best parameters are ", str(grid.best_params_), "score", grid.best_score_
        print "error", berrors1['err'], "sen ", berrors1['sen'], "spe ", berrors1['spe'], "auc ", berrors1['auc'], "mcc1 ", berrors1['mcc']

    if deb!=0:
        print "saving the model"
    joblib.dump(grid, model_output_path)

    return grid
