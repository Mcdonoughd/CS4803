from problem1 import *
import numpy as np
from sklearn.datasets import make_classification

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_z():

    # an example feature vector with 2 features
    x = np.array([1., 2.])
    
    w = np.array([0.5,-0.6])
    b = 0.2

    z = compute_z(x,w,b)

    assert type(z) == float
    assert np.allclose(z, -0.5, atol = 1e-3) 

    x = np.array([2., 5.])
    z = compute_z(x,w,b)

    assert np.allclose(z, -1.8, atol = 1e-3) 



#-------------------------------------------------------------------------
def test_compute_a():
    a =compute_a(0.)
    assert type(a) == float
    assert np.allclose(a, 0.5, atol = 1e-2) 

    a =compute_a(1.)
    assert np.allclose(a, 0.73105857863, atol = 1e-2) 

    a = compute_a(-1.)
    assert np.allclose(a, 0.26894142137, atol = 1e-2) 

    a =compute_a(-100.)
    assert np.allclose(a, 0, atol = 1e-2) 

    a =compute_a(100.)
    assert np.allclose(a, 1, atol = 1e-2) 

#-------------------------------------------------------------------------
def test_compute_L():
    ''' test the correctness of cross_entropy() function in problem1.py'''
    
    L= compute_L(1.,1)

    assert type(L) == float
    assert np.allclose(L, 0., atol = 1e-3) 

    L= compute_L(0.,0)
    assert np.allclose(L, 0., atol = 1e-3) 

    L= compute_L(0.5,1)
    assert np.allclose(L, 0.69314718056, atol = 1e-3) 

    L= compute_L(0.5,0)
    assert np.allclose(L, 0.69314718056, atol = 1e-3) 

#-------------------------------------------------------------------------
def test_forward():
    ''' test the correctness of forward() function in problem1.py'''
    x = np.array([1., 2.])
    w = np.array([0., 0.])
    b = 0.
    y = 1 
    z, a, L= forward(x,y,w,b)
    z_true, a_true, L_true = 0.0,0.5,0.69314718056
    assert np.allclose(z,z_true, atol=1e-3)
    assert np.allclose(a,a_true, atol=1e-3)
    assert np.allclose(L,L_true, atol=1e-3)


#-------------------------------------------------------------------------
def test_compute_dL_da():
    a  = 0.5 
    y = 1
    dL_da = compute_dL_da(a,y)

    assert type(dL_da) == float 
    assert np.allclose(dL_da, -2., atol= 1e-3)

    a  = 0.5 
    y = 0
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, 2., atol= 1e-3)

    a  = 0.9 
    y = 0
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, 10., atol= 1e-3)

#-------------------------------------------------------------------------
def test_check_dL_da():
    a = max(np.random.random(1),1e-7)
    y = 1 
    # analytical gradients
    da = compute_dL_da(a,y)
    # numerical gradients
    da_true = check_dL_da(a,y)
    assert np.allclose(da, da_true, atol= 1e-3)

    #-----------------------
    a = max(np.random.random(1), 1e-7)
    y = 0 
    # analytical gradients
    da = compute_dL_da(a,y)
    # numerical gradients
    da_true = check_dL_da(a,y)
    assert np.allclose(da, da_true, atol= 1e-3)

#-------------------------------------------------------------------------
def test_compute_da_dz():
    a  = 0.3 
    da_dz = compute_da_dz(a)

    assert type(da_dz) == float 
    assert np.allclose(da_dz, 0.21, atol= 1e-3)

    a  = 0.5 
    da_dz = compute_da_dz(a)
    assert np.allclose(da_dz, 0.25, atol= 1e-3)

    a  = 0.9 
    da_dz = compute_da_dz(a)
    assert np.allclose(da_dz, 0.09, atol= 1e-3)

    a  = 0.01
    da_dz = compute_da_dz(a)
    assert np.allclose(da_dz, 0.0099, atol= 1e-4)


#-------------------------------------------------------------------------
def test_check_da_dz():
    z = np.random.random(1)
    a = compute_a(z)
    # analytical gradients
    da_dz = compute_da_dz(a)
    # numerical gradients
    da_dz_true = check_da_dz(z)
    assert np.allclose(da_dz, da_dz_true, atol=1e-4) 



#-------------------------------------------------------------------------
def test_compute_dz_dw():

    x = np.array([1., 2.])
    dz_dw = compute_dz_dw(x)

    assert type(dz_dw) == np.ndarray
    assert dz_dw.shape == (2,) 

    dz_dw_true = np.array([1., 2.])
    dz_db_true = 1.
    assert np.allclose(dz_dw, dz_dw_true, atol=1e-2) 

#-------------------------------------------------------------------------
def test_check_dz_dw():
    x = np.random.random(5)
    w = np.random.random(5)
    b = np.random.random(1)

    # analytical gradients
    dw = compute_dz_dw(x)
    # numerical gradients
    dw_true = check_dz_dw(x,w,b, delta=10)

    assert np.allclose(dw, dw_true, atol=1e-2) 

#-------------------------------------------------------------------------
def test_compute_dz_db():
    dz_db = compute_dz_db()
    assert type(dz_db) == float 
    dz_db_true = 1.
    assert np.allclose(dz_db, dz_db_true, atol=1e-2) 

#-------------------------------------------------------------------------
def test_check_dz_db():
    x = np.random.random(5)
    w = np.random.random(5)
    b = np.random.random(1)

    # analytical gradients
    db = compute_dz_db()
    # numerical gradients
    db_true = check_dz_db(x,w,b, delta=10)

    assert np.allclose(db, db_true, atol=1e-2) 

#-------------------------------------------------------------------------
def test_backward():
    ''' test the correctness of backward() function in problem1.py'''
    x = np.array([1., 2.])
    y = 1 
    a = 0.5
    da, dz, dw, db = backward(x,y,a)
    da_true = -2.0 
    dz_true = 0.25 
    dw_true = np.array([ 1., 2.])
    db_true = 1.0
    assert np.allclose(da,da_true, atol=1e-3)
    assert np.allclose(dz,dz_true, atol=1e-3)
    assert np.allclose(dw,dw_true, atol=1e-3)
    assert np.allclose(db,db_true, atol=1e-3)


#-------------------------------------------------------------------------
def test_compute_dL_dw():
    dL_da = -2.0 
    da_dz = 0.25 
    dz_dw = np.array([ 1., 2.])
    dz_db = 1.0

    dL_dw = compute_dL_dw(dL_da,da_dz, dz_dw) 
    
    assert type(dL_dw) == np.ndarray
    assert dL_dw.shape == (2,) 

    dL_dw_true = [-0.5, -1.]
    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
    
#-------------------------------------------------------------------------
def test_check_dL_dw():
    x = np.random.random(6)
    y = np.random.randint(0,2) 
    w = np.random.random(6)
    b = np.random.random(1)

    z, a, L= forward(x,y,w,b)
    dL_da, da_dz, dz_dw, dz_db = backward(x,y,a)

    # analytical gradients
    dL_dw = compute_dL_dw(dL_da, da_dz, dz_dw)
    # numerical gradients
    dL_dw_true = check_dL_dw(x,y,w,b)

    assert np.allclose(dL_dw, dL_dw_true, atol = 1e-3)
 
#-------------------------------------------------------------------------
def test_compute_dL_db():
    dL_da = -2.0 
    da_dz = 0.25 
    dz_db = 1.0

    dL_db = compute_dL_db(dL_da,da_dz,dz_db)
    
    assert type(dL_db) == float 

    dL_db_true = -0.5
    assert np.allclose(dL_db, dL_db_true, atol = 1e-3)
   
#-------------------------------------------------------------------------
def test_check_dL_db():
    x = np.random.random(6)
    w = np.random.random(6)
    b = np.random.random(1)
    y = np.random.randint(0,2) 
    z, a, L= forward(x,y,w,b)
    dL_da, da_dz, dz_dw, dz_db = backward(x,y,a)

    # analytical gradients
    dL_db = compute_dL_db(dL_da, da_dz, dz_db)
    # numerical gradients
    dL_db_true = check_dL_db(x,y,w,b)
    assert np.allclose(dL_db, dL_db_true, atol = 1e-3)

#-------------------------------------------------------------------------
def test_update_w():

    w = np.array( [0., 0.])
    dL_dw = np.array( [1., 2.])

    w = update_w(w,dL_dw, alpha=.5) 
    
    w_true = - np.array([0.5, 1.])
    assert np.allclose(w, w_true, atol = 1e-3)

    w = update_w(w,dL_dw, alpha=1.) 
    w_true = - np.array([1.5, 3.])
    assert np.allclose(w, w_true, atol = 1e-3)

#-------------------------------------------------------------------------
def test_update_b():
    b = 0.
    dL_db = 2. 

    b = update_b(b, dL_db, alpha=.5) 
    
    b_true = -1. 
    assert np.allclose(b, b_true, atol = 1e-3)


    b = update_b(b, dL_db, alpha=1.) 
    b_true = -3.
    assert np.allclose(b, b_true, atol = 1e-3)



#-------------------------------------------------------------------------
def test_train():
    ''' test the correctness of train() function in problem1.py'''
    # an example feature matrix (4 instances, 2 features)
    Xtrain  = np.array( [[0., 1.],
                         [1., 0.],
                         [0., 0.],
                         [1., 1.]])
    Ytrain = np.array([0, 1, 0, 1])

    # call the function
    w, b = train(Xtrain, Ytrain, alpha=1., n_epoch = 100)
   
    assert w[0]+w[1] + b > 0 # x4 is positive
    assert w[0] + b > 0 # x2 is positive
    assert w[1] + b < 0 # x1 is negative 
    assert  b < 0 # x3 is negative 

    #------------------
    # another example
    Ytrain = np.array([1, 0, 0, 1])
    w, b = train(Xtrain, Ytrain, alpha=0.01, n_epoch = 10)
    assert w[0]+w[1] + b > 0 # x4 is positive
    assert w[0] + b < 0 # x2 is positive
    assert w[1] + b > 0 # x1 is negative 
    assert  b < 0 # x3 is negative 

    #------------------
    # another example
    Xtrain  = np.array( [[0., 1.],
                         [1., 0.],
                         [0., 0.],
                         [2., 0.],
                         [0., 2.],
                         [1., 1.]])
    Ytrain = np.array([0, 0,0,1, 1, 1])
    w, b = train(Xtrain, Ytrain, alpha=0.1, n_epoch = 1000)
    assert w[0]+w[1] + b > 0 
    assert 2*w[0] + b > 0 
    assert 2*w[1] + b > 0 
    assert w[0] + b < 0 
    assert w[1] + b < 0 
    assert  b < 0 
  
#-------------------------------------------------------------------------
def test_predict():
    ''' test the correctness of predict() function in problem1.py'''

    # an example feature matrix (4 instances, 2 features)
    Xtest  = np.array( [ [0., 1.],
                         [1., 0.],
                         [2., 2.],
                         [1., 1.]])
    
    w = np.array( [ 0.5, -0.6])
    b = 0.2

    # call the function
    Ytest, Ptest = predict(Xtest, w, b )

    assert type(Ytest) == np.ndarray 
    assert Ytest.shape == (4,) 
    assert type(Ptest) == np.ndarray 
    assert Ptest.shape == (4,) 

    Ytest_true = [0, 1,1, 1]
    Ptest_true = [[0.401312339887548, 0.6681877721681662, 0.5, 0.52497918747894]]

    # check the correctness of the result 
    assert np.allclose(Ytest, Ytest_true, atol = 1e-2)
    assert np.allclose(Ptest, Ptest_true, atol = 1e-2)



#-------------------------------------------------------------------------
def test_logistic_regression():
    ''' test the correctness of both train() and predict() functionin problem1.py'''
    # create a binary classification dataset
    n_samples = 200
    X,y = make_classification(n_samples= n_samples,
                              n_features=4, n_redundant=0, n_informative=3,
                              n_classes= 2,
                              class_sep = 1.,
                              random_state=1)
        
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    w,b = train(Xtrain, Ytrain,alpha=.001, n_epoch=1000)
    Y, P = predict(Xtrain, w, b)
    accuracy = sum(Y == Ytrain)/(n_samples/2.)
    print 'Training accuracy:', accuracy
    assert accuracy > 0.9
    Y, P = predict(Xtest, w, b)
    accuracy = sum(Y == Ytest)/(n_samples/2.)
    print 'Test accuracy:', accuracy
    assert accuracy > 0.9

