from problem3 import *
import numpy as np
from sklearn.datasets import make_classification

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_a1():
    z1 = np.array([0.,1.])
    a1 = compute_a1(z1)
    assert type(a1) == np.ndarray 
    assert a1.shape == (2,)
    assert np.allclose(a1, [0.5,0.731], atol = 1e-3) 


    z1 = np.array([-1.,-100., 100])
    a1 = compute_a1(z1)
    assert a1.shape == (3,)
    assert np.allclose(a1, [0.2689, 0, 1], atol = 1e-2) 

#-------------------------------------------------------------------------
def test_forward():
    ''' test the correctness of forward() function'''
    x = np.array([1., 2.,3.,4])

    # first layer with 3 neurons
    W1 = np.array([[0.,0.,0.,0.],
                   [0.,0.,0.,0.],
                   [0.,0.,0.,0.]])
    b1 = np.array([0.,0.,0.])

    # second layer with 2 neurons
    W2 = np.array([[0.,0.,0.],
                   [0.,0.,0.]])
    b2 = np.array([100.,0.])

    z1, a1, z2, a2 = forward(x,W1,b1,W2,b2) 
    
    assert type(z1) == np.ndarray
    assert type(a1) == np.ndarray
    assert z1.shape == (3,)
    assert a1.shape == (3,)
    assert type(z2) == np.ndarray
    assert type(a2) == np.ndarray
    assert z2.shape == (2,)
    assert a2.shape == (2,)

    assert np.allclose(z1, [0,0,0], atol = 1e-3)
    assert np.allclose(a1, [0.5,0.5,0.5], atol = 1e-3)
    assert np.allclose(z2, [100,0], atol = 1e-3)
    assert np.allclose(a2, [1,0], atol = 1e-3)


#-------------------------------------------------------------------------
def test_compute_dz2_da1():
    W2= np.array([[0.,0.,3.,0.],
                  [0.,1.,0.,0.],
                  [0.,0.,0.,5.]])
    dz2_da1 = compute_dz2_da1(W2)

    assert type(dz2_da1) == np.ndarray
    assert dz2_da1.shape == (3,4)
    assert np.allclose(dz2_da1, W2, atol= 1e-3)

#-------------------------------------------------------------------------
def test_compute_da1_dz1():
    a1= np.array([.5,.5,.3,.6])
    da1_dz1 = compute_da1_dz1(a1)

    assert type(da1_dz1) == np.ndarray
    assert da1_dz1.shape == (4,)
    assert np.allclose(da1_dz1, [.25,.25,.21,.24], atol= 1e-3)

#-------------------------------------------------------------------------
def test_backward():
    ''' test the correctness of backward() function'''
    x = np.array([1., 2.,3.,4])
    y = 1

    # first layer with 3 hidden neurons
    W1 = np.array([[0.,0.,0.,0.],
                   [0.,0.,0.,0.],
                   [0.,0.,0.,0.]])
    b1 = np.array([0.,0.,0.])

    # second layer with 2 hidden neurons
    W2 = np.array([[0.,0.,0.],
                   [0.,0.,0.]])
    b2 = np.array([0.,0.])

    z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)

    dL_da2, da2_dz2, dz2_dW2, dz2_db2, dz2_da1, da1_dz1, dz1_dW1, dz1_db1= backward(x,y,a1,a2, W2) 
    
    assert type(dL_da2) == np.ndarray 
    assert dL_da2.shape == (2,)
    np.allclose(dL_da2,[0.,-2.],atol=1e-3)

    assert type(da2_dz2) == np.ndarray 
    assert da2_dz2.shape == (2,2)
    np.allclose(da2_dz2,[[.25,-.25],[-.25,.25]],atol=1e-3)

    assert type(dz2_dW2) == np.ndarray 
    assert dz2_dW2.shape == (2,3)
    np.allclose(dz2_dW2,[[.5,.5,.5],[.5,.5,.5]],atol=1e-3)

    assert type(dz2_db2) == np.ndarray 
    assert dz2_db2.shape == (2,)
    np.allclose(dz2_db2,[1,1],atol=1e-3)

    assert type(dz2_da1) == np.ndarray 
    assert dz2_da1.shape == (2,3)
    t = [[ 0., 0., 0.],
         [ 0., 0., 0.]]
    np.allclose(dz2_da1,t,atol=1e-3)

    assert type(da1_dz1) == np.ndarray 
    assert da1_dz1.shape == (3,)
    np.allclose(da1_dz1,[.25,.25,.25],atol=1e-3)

    assert type(dz1_dW1) == np.ndarray 
    assert dz1_dW1.shape == (3,4)
    t = [[ 1.,  2.,  3.,  4.],
         [ 1.,  2.,  3.,  4.],
         [ 1.,  2.,  3.,  4.]] 

    assert type(dz1_db1) == np.ndarray 
    assert dz1_db1.shape == (3,)
    np.allclose(dz1_db1,[1,1,1],atol=1e-3)

##-------------------------------------------------------------------------
def test_compute_gradients():
    x = np.array([1., 2.,3.,4])
    y = 1

    # first layer with 3 hidden neurons
    W1 = np.array([[0.,0.,0.,0.],
                   [0.,0.,0.,0.],
                   [0.,0.,0.,0.]])
    b1 = np.array([0.,0.,0.])

    # second layer with 2 hidden neurons
    W2 = np.array([[0.,0.,0.],
                   [0.,0.,0.]])
    b2 = np.array([0.,0.])

    # forward pass
    z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
    print 'a1:', a1 

    # backward pass: prepare local gradients
    dL_da2, da2_dz2, dz2_dW2, dz2_db2, dz2_da1, da1_dz1, dz1_dW1, dz1_db1= backward(x,y,a1,a2, W2) 
    # call the function 
    dL_dW2, dL_db2, dL_dW1, dL_db1 = compute_gradients(dL_da2, da2_dz2, dz2_dW2, dz2_db2, dz2_da1, da1_dz1, dz1_dW1, dz1_db1)
    
    assert type(dL_dW2) == np.ndarray 
    assert dL_dW2.shape == (2,3)
    t = [[ 0.25, 0.25, 0.25],
         [-0.25,-0.25,-0.25]]
    np.allclose(dL_dW2,t,atol=1e-3)
 
    assert type(dL_db2) == np.ndarray 
    assert dL_db2.shape == (2,)
    t = [0.5,-0.5]
    np.allclose(dL_db2,t,atol=1e-3)

    assert type(dL_dW1) == np.ndarray 
    assert dL_dW1.shape == (3,4)
    t = np.zeros((3,4)) 
    np.allclose(dL_dW1,t,atol=1e-3)

    assert type(dL_db1) == np.ndarray 
    assert dL_db1.shape == (3,)
    t = [0,0,0]
    np.allclose(dL_db1,t,atol=1e-3)


##-------------------------------------------------------------------------
def test_check_compute_gradients():
    p = np.random.randint(2,10) # number of features
    c = np.random.randint(2,10) # number of classes
    h = np.random.randint(2,10) # number of neurons in the 1st layer 
    x = np.random.random(p)
    y = np.random.randint(c) 
    W1 = np.random.random((h,p))
    b1 = np.random.random(h)
    W2 = np.random.random((c,h))
    b2 = np.random.random(c)
    z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
    dL_da2, da2_dz2, dz2_dW2, dz2_db2, dz2_da1, da1_dz1, dz1_dW1, dz1_db1= backward(x,y,a1,a2, W2) 

    # analytical gradients
    dL_dW2, dL_db2, dL_dW1, dL_db1 = compute_gradients(dL_da2, da2_dz2, dz2_dW2, dz2_db2, dz2_da1, da1_dz1, dz1_dW1, dz1_db1)
    # numerical gradients
    dL_dW2_true = check_dL_dW2(x,y, W1,b1,W2,b2)
    assert np.allclose(dL_dW2, dL_dW2_true, atol=1e-4) 

    dL_dW1_true = check_dL_dW1(x,y, W1,b1,W2,b2)
    print dL_dW1_true
    assert np.allclose(dL_dW1, dL_dW1_true, atol=1e-4) 

#-------------------------------------------------------------------------
def test_fully_connected():
    ''' test the correctness of both train() and predict() function'''
    # create a multi-class classification dataset
    n_samples = 200
    X,y = make_classification(n_samples= n_samples,
                              n_features=5, n_redundant=0, n_informative=4,
                              n_classes= 3,
                              class_sep = 5.,
                              random_state=1)
        
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    W1,b1,W2,b2 = train(Xtrain, Ytrain,alpha=.01, n_epoch=100)
    Y, P = predict(Xtrain, W1, b1, W2, b2)
    accuracy = sum(Y == Ytrain)/(n_samples/2.)
    print 'Training accuracy:', accuracy
    assert accuracy > 0.9
    Y, P = predict(Xtest, W1, b1, W2, b2)
    accuracy = sum(Y == Ytest)/(n_samples/2.)
    print 'Test accuracy:', accuracy
    assert accuracy > 0.9

