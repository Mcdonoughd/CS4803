from problem1 import *
import numpy as np
from sklearn.datasets import make_classification

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_distance():
    ''' test the correctness of compute_distance() function in problem1.py'''

    # an example feature matrix (3 instances, 2 features)
    Xtrain = np.array( [ [0., 1.],
                         [1., 0.],
                         [1., 1.]])

    # an example test dataset (2 instances, 2 features)
    Xtest = np.array( [ [0., 1.],
                         [1., 1.]])

    # call the function
    D = compute_distance(Xtrain, Xtest)

    # test whether or not the result is a float number 
    assert type(D) == np.ndarray 

    D_true = np.array( [ [0., 1.41421356, 1.],
                         [1., 1.        , 0.]])

    # check the correctness of the result 
    assert np.allclose(D, D_true, atol = 1e-4)

    #-------------------------
    # another example with 2 instances 3 features. 

    Xtrain = np.array( [ [0., 1., 1.],
                         [1., 1., 1.]])
    Xtest = np.array( [ [0., 1., 1.],
                        [1., 1., 0.]])

    # call the function
    D = compute_distance(Xtrain, Xtest)

    D_true = np.array( [ [0.        , 1.],
                         [1.41421356, 1.]])

    # check the correctness of the result 
    assert np.allclose(D, D_true, atol = 1e-4)



#-------------------------------------------------------------------------
def test_k_nearest_neighbor():
    ''' test the correctness of k_nearest_neighbor() function in problem1.py'''

    # an example feature matrix (3 instances, 2 features)
    Xtrain = np.array( [ [0., 1.],
                         [1., 0.],
                         [1., 1.]])
    Ytrain = [1, 2, 3]
    # an example test dataset (2 instances, 2 features)
    Xtest = np.array( [ [0., 1.],
                         [1., 1.]])

    # call the function
    Ytest = k_nearest_neighbor(Xtrain, Ytrain,Xtest, K=1)

    # test whether or not the result is a float number 
    assert type(Ytest) == np.ndarray 

    Ytest_true=[1,3] 

    # check the correctness of the result 
    assert np.allclose(Ytest, Ytest_true, atol = 1e-4)
    #-------------------------
    # another example 

    Xtrain = np.array( [ [0., 1.],
                         [1., 0.],
                         [1., 1.]])
    Ytrain = [1, 2, 2]
    # an example test dataset (2 instances, 2 features)
    Xtest = np.array( [ [0., 1.],
                         [1., 1.]])

    # call the function
    Ytest = k_nearest_neighbor(Xtrain, Ytrain, Xtest)

    # test whether or not the result is a float number 
    Ytest_true= [2, 2]

    # check the correctness of the result 
    assert np.allclose(Ytest, Ytest_true, atol = 1e-4)


#-------------------------------------------------------------------------
def test_k_nearest_neighbor_toydata():
    ''' test the correctness of k_nearest_neighbor() function in problem1.py'''
    # create a classification dataset
    X,y = make_classification(n_samples= 400,
                              n_features=2, n_redundant=0, n_informative=2,
                              n_classes = 3,
                              class_sep = 2.,
                              random_state=1, n_clusters_per_class=1)
        
    # split into a training set and a test set
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    Y = k_nearest_neighbor(Xtrain, Ytrain,Xtest, K=3)

    # compute the accuracy of classification on test set
    accuracy = sum(Y == Ytest)/200.
    print 'classification accuracy:',accuracy
    assert accuracy > 0.8




