# SVM-and-Softmax-Classifier
Understanding the working of SVM and Softmax Classifier.


Instructions:
Before you start, install matplotlib package to plot images and weights, and install Keras
to load CIFAR10 dataset by using this command.

- Windows10:
o Download “scipy-0.19.1-cp35-cp35m-win_amd64.whl” from
http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
o Go to download location of file using cmd then type
o “pip3 install scipy-0.19.1-cp35-cp35m-win_amd64.whl”
o “pip3 install matplotlib==2.0.0”
o “pip3 install --upgrade keras”

- Linux/Mac:
o Open terminal and run
o “sudo pip3 install matplotlib==2.0.0”
o “sudo pip3 install --upgrade keras”

About the files:
- runSvmSoftmax.py: This is the main file that you will execute. It will read and
processing CIFAR10 dataset, initialize classifiers, training, and also tune up hyper
parameters.
- svm.py: SVM class that contains 5 functions: initialize, train, predict, calculate
loss, and calculate accuracy.
- softmax.py: Softmax class that has the same structure as in svm.py.

-- Run "runSvmSoftmax.py" to get the output.
