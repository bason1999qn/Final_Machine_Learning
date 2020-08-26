from sklearn.svm import SVC
import pickle
import gzip
from sklearn.metrics import accuracy_score
from datetime import datetime
from threading import Thread

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

   train_X, train_Y = train_data
   val_X, val_Y = val_data
   test_X, test_Y = test_data

   return train_X, train_Y, val_X, val_Y, test_X, test_Y

# Test
train_X, train_Y, val_X, val_Y, test_X, test_Y = read_mnist('mnist.pkl.gz')

def test_train(C, gamma):
    # Training:
    start_time = datetime.now()
    svclassifier = SVC(C=C, gamma=gamma, kernel='rbf')
    svclassifier.fit(train_X, train_Y)
    train_time = datetime.now() - start_time

    # Get Y predict from input X
    y_train_pred = svclassifier.predict(train_X)
    y_val_pred = svclassifier.predict(val_X)

    # Get error:
    train_error = 1 - (accuracy_score(train_Y, y_train_pred, normalize=False) / float(train_Y.size))
    val_error = 1 - (accuracy_score(val_Y, y_val_pred, normalize=False) / float(val_Y.size))

    # Print results:
    print("C: {}  |  gamma: {}  |  train error: {}  |  validation error: {}  |  time: {}"
          .format(C, gamma, train_error, val_error, train_time))


#Link: https://stackoverflow.com/questions/51814897/how-to-process-a-list-in-parallel-in-python
# ----------------------------------------------



C_range = [0.01, 0.1, 1, 10, 100]
gamma_range = [0.0001, 0.001, 0.01, 0.1, 1]

for C in C_range:
    for gamma in gamma_range:
        t = Thread(target=test_train, args=(C, gamma))
        t.start()



