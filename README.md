# 2018_occupancy
Files description
-- main.py Executes nn_and_bayes_error_sub.py and svm_error.py
-- nn_and_bayes_error_sub.py computes the classification error using NN algorithm along with Bayes limits.
-- svm_error.py computes the classification error using an SVM with RBF kernel.
-- svm_training.py its an implementation of the SVM algorithm using kernel functions.
-- myfuncs.py computes error indices and does a min-max normalization

--------------------------------------
List of requirements and dependencies
--------------------------------------
Requirement: numpy
from https://docs.scipy.org/doc/numpy-1.13.0/user/install.html

Installation via zipper
$ install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
or installtion via pip
$ python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
--------------------------------------
Requirement: tensor flow with cpu
from: https://www.tensorflow.org/install/install_linux
Install pip
$ sudo apt-get install python-pip python-dev   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev # for Python 3.n

Check if you are using pip(python3) or pip2(python2)
$ sudo pip2 install --upgrade pip
$ sudo pip install --upgrade pip

Also install openpyxl 2.2.4 and html5lib

Installation of tensorflow
$ pip install tensorflow      # Python 2.7; CPU support (no GPU support)
$ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)

Some versions of tensorflow have memory dump problems, so you can try ver 1.5
$ pip install tensorflow==1.5      # Python 2.7; CPU support (no GPU support)
$ pip2 install tensorflow==1.5     # Python 3.n; CPU support (no GPU support)
--------------------------------------
Requirement: scipy
$ pip2 install scipy
--------------------------------------
Requirement: scikit-learn
$ pip2 install -U scikit-learn
$ pip install -U scikit-learn
