from sklearn.svm import SVC
import numpy as np
import pickle
import gzip
import pandas as pd
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


def read_mnist(mnist_file):
    """
    Reads MNIST data.

    Parameters
    ----------
    mnist_file : string
        The name of the MNIST file (e.g., 'mnist.pkl.gz').

    Returns
    -------
    (train_X, train_Y, val_X, val_Y, test_X, test_Y) : tuple
        train_X : numpy array, shape (N=50000, d=784)
            Input vectors of the training set.
        train_Y: numpy array, shape (N=50000)
            Outputs of the training set.
        val_X : numpy array, shape (N=10000, d=784)
            Input vectors of the validation set.
        val_Y: numpy array, shape (N=10000)
            Outputs of the validation set.
        test_X : numpy array, shape (N=10000, d=784)
            Input vectors of the test set.
        test_Y: numpy array, shape (N=10000)
            Outputs of the test set.
    """
    f = gzip.open(mnist_file, 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')

train_X, train_Y = train_data
val_X, val_Y = val_data
test_X, test_Y = test_data

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(train_X, train_Y)
y_pred = svclassifier.predict(test_X)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_Y, y_pred))
print(classification_report(test_Y, y_pred))

list_error = []
for k in range(-4,4):
    t = 10**k
    start_time = time.perf_counter()
    svclassifier = SVC(C = 10**k,kernel='rbf')
    svclassifier.fit(train_X, train_Y)
    train_time = time.perf_counter() - start_time
    y_pred = svclassifier.predict(test_X)
    my_accuracy = accuracy_score(test_Y, y_pred, normalize=False) / float(test_Y.size)
    error = 1 - my_accuracy
    print("C: {},error: {}, time: {}".format(t, error, train_time))
    list_error.append(error)


