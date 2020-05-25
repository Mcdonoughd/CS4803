from problem2 import *
import numpy as np
from sklearn.datasets import make_classification

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_z():
    x = np.array([1., 2., 3.])
    
    W = np.array([[0.5,-0.6,0.3],
                  [0.6,-0.5,0.2]])
    b = np.array([0.2, 0.3])

    z = compute_z(x,W,b)

    assert type(z) == np.ndarray 
    assert z.shape == (2,)
    assert np.allclose(z, [0.4,0.5], atol = 1e-3) 

    x = np.array([2., 5.,2.])
    z = compute_z(x,W,b)

    assert np.allclose(z, [-1.2,-0.6], atol = 1e-3) 

#-------------------------------------------------------------------------
def test_compute_a():
    a = compute_a([1., 1.])
    assert type(a) == np.ndarray 
    assert np.allclose(a, [0.5, 0.5], atol = 1e-2) 

    a = compute_a([1., 1.,1., 1.])
    assert np.allclose(a, [0.25, 0.25, 0.25, 0.25], atol = 1e-2) 


    a = compute_a([-1., -1.,-1., -1.])
    assert np.allclose(a, [0.25, 0.25, 0.25, 0.25], atol = 1e-2) 


    a = compute_a([-2., -1.,1., 2.])
    assert np.allclose(a, [ 0.01275478,0.03467109,0.25618664,0.69638749], atol = 1e-2)
    
#-------------------------------------------------------------------------
def test_compute_L():
    
    L = compute_L([0.,1.],1)

    assert type(L) == float
    assert np.allclose(L, 0., atol = 1e-3) 

    L = compute_L([0.,0.,1.], 2)
    assert np.allclose(L, 0., atol = 1e-3) 

    L= compute_L([1.,0.], 0)
    assert np.allclose(L, 0., atol = 1e-3) 

    L= compute_L([.5,.5], 0)
    assert np.allclose(L, 0.69314718056, atol = 1e-3) 

    L= compute_L([.5,.5], 1)
    assert np.allclose(L, 0.69314718056, atol = 1e-3) 

#-------------------------------------------------------------------------
def test_forward():
    ''' test the correctness of forward() function'''
    x = np.array([1., 2.,3.])
    W = np.array([[0., 0.,0.],[0.,0.,0.]])
    b = np.array([0.,0.])
    y = 1

    z, a, L= forward(x,y,W,b) 
    
    assert type(z) == np.ndarray
    assert type(a) == np.ndarray 
    assert z.shape == (2,) 
    assert a.shape == (2,) 

    z_true = [0.,0.]
    a_true = [0.5,0.5]
    L_true = 0.69314718056  
    assert np.allclose(z, z_true, atol = 1e-3)
    assert np.allclose(a, a_true, atol = 1e-3)
    assert np.allclose(L, L_true, atol = 1e-3)

#-------------------------------------------------------------------------
def test_compute_dL_da():
    a  = np.array([0.5,0.5])
    y = 1
    dL_da = compute_dL_da(a,y)

    assert type(dL_da) == np.ndarray 
    assert dL_da.shape == (2,) 
    assert np.allclose(dL_da, [0.,-2.], atol= 1e-3)

    a  = np.array([0.5,0.5])
    y = 0
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, [-2.,0.], atol= 1e-3)

    a  = np.array([0.1,0.6,0.1,0.2])
    y = 3
    dL_da = compute_dL_da(a,y)
    assert np.allclose(dL_da, [0.,0.,0.,-5.], atol= 1e-3)


#-------------------------------------------------------------------------
def test_check_dL_da():
    c = np.random.randint(2,10) # number of classes
    a  = np.random.random(c) # activation
    y = np.random.randint(c) # label 
    # analytical gradients
    da = compute_dL_da(a,y)
    # numerical gradients
    da_true = check_dL_da(a,y)
    assert np.allclose(da, da_true, atol= 1e-3)



#-------------------------------------------------------------------------
def test_compute_da_dz():

    a  = np.array([0.3, 0.7])
    da_dz = compute_da_dz(a)

    assert type(da_dz) == np.ndarray 
    assert da_dz.shape == (2,2)

    assert np.allclose(da_dz, [[.21,-.21],[-.21,.21]], atol= 1e-3)

    a  = np.array([0.1, 0.2, 0.7])
    da_dz = compute_da_dz(a)
    assert da_dz.shape == (3,3)

    da_dz_true = np.array( [[ 0.09, -0.02, -0.07],
                         [-0.02,  0.16, -0.14],
                         [-0.07, -0.14,  0.21]])

    assert np.allclose(da_dz,da_dz_true,atol= 1e-3)



#-------------------------------------------------------------------------
def test_check_da_dz():
    z = np.random.random(6)
    a  = compute_a(z)
    # analytical gradients
    dz = compute_da_dz(a)
    # numerical gradients
    dz_true = check_da_dz(z)
    assert np.allclose(dz, dz_true, atol= 1e-3)



#-------------------------------------------------------------------------
def test_compute_dz_dW():
    x = np.array([1., 2.,3.])
    dz_dW = compute_dz_dW(x,2)

    assert type(dz_dW) == np.ndarray
    assert dz_dW.shape == (2,3) 

    dz_dW_true = np.array([[1., 2.,3],[1., 2.,3]])
    assert np.allclose(dz_dW, dz_dW_true, atol=1e-2) 

#-------------------------------------------------------------------------
def test_check_dz_dW():
    x = np.random.random(5)
    W = np.random.random((3,5))
    b = np.random.random(3)
    # analytical gradients
    dz_dW = compute_dz_dW(x,3)
    # numerical gradients
    dz_dW_true = check_dz_dW(x,W,b)

    assert np.allclose(dz_dW, dz_dW_true, atol=1e-3) 

#-------------------------------------------------------------------------
def test_compute_dz_db():
    dz_db = compute_dz_db(2)

    assert type(dz_db) == np.ndarray
    assert dz_db.shape == (2,) 

    dz_db_true = np.array([1.,1.])
    assert np.allclose(dz_db, dz_db_true, atol=1e-2) 


#-------------------------------------------------------------------------
def test_check_dz_db():
    x = np.random.random(5)
    W = np.random.random((3,5))
    b = np.random.random(3)
    # analytical gradients
    dz_db = compute_dz_db(3)
    # numerical gradients
    dz_db_true = check_dz_db(x,W,b)

    assert np.allclose(dz_db, dz_db_true, atol=1e-3) 

#-------------------------------------------------------------------------
def test_backward():
    ''' test the correctness of backward() function'''
    x = np.array([1., 2.,3.])
    y = 1
    a = np.array([.5, .5])

    da, dz, dW, db = backward(x,y,a) 
    
    assert type(da) == np.ndarray
    assert type(dz) == np.ndarray 
    assert type(dW) == np.ndarray 
    assert type(db) == np.ndarray 
    assert da.shape == (2,) 
    assert dz.shape == (2,2) 
    assert dW.shape == (2,3) 
    assert db.shape == (2,) 

    da_true = [0.,-2.]
    dz_true = [[0.25, -0.25],[-0.25,0.25]]
    dW_true = [[ 1., 2., 3.], [ 1., 2., 3.]] 
    db_true = [1.,1.]

    assert np.allclose(da, da_true, atol = 1e-3)
    assert np.allclose(dz, dz_true, atol = 1e-3)
    assert np.allclose(dW, dW_true, atol = 1e-3)
    assert np.allclose(db, db_true, atol = 1e-3)
    

#-------------------------------------------------------------------------
def test_compute_dL_dz():
    dL_da = np.array([0.,-2.])
    da_dz = np.array([[0.25, -0.25],[-0.25,0.25]])
    dz_dW = np.array([[ 1., 2., 3.], [ 1., 2., 3.]])

    dL_dz = compute_dL_dz(dL_da, da_dz) 
    
    assert type(dL_dz) == np.ndarray
    assert dL_dz.shape == (2,) 
    print dL_dz

    dL_dz_true = [0.5, -0.5]
    assert np.allclose(dL_dz, dL_dz_true, atol = 1e-3)


#-------------------------------------------------------------------------
def test_compute_dL_dW():
    dL_dz = np.array([0.5, -0.5])
    dz_dW = np.array([[ 1., 2., 3.], [ 1., 2., 3.]] )

    dL_dW = compute_dL_dW(dL_dz, dz_dW) 
    
    assert type(dL_dW) == np.ndarray
    assert dL_dW.shape == (2,3) 
    print dL_dW
    dL_dW_true = [[ 0.5,  1.0,  1.5],
               [-0.5, -1.0, -1.5]]
    assert np.allclose(dL_dW, dL_dW_true, atol = 1e-3)

#-------------------------------------------------------------------------
def test_check_dL_dW():
    p = np.random.randint(2,10) # number of features
    c = np.random.randint(2,10) # number of classes
    x = np.random.random(p)
    y = np.random.randint(c) 
    W = np.random.random((c,p))
    b = np.random.random(c)
    z, a, L = forward(x,y,W,b)
    dL_da, da_dz, dz_dW, dz_db = backward(x,y,a)
    dL_dz = compute_dL_dz(dL_da, da_dz)

    # analytical gradients
    dL_dW = compute_dL_dW(dL_dz,dz_dW)
    # numerical gradients
    dL_dW_true = check_dL_dW(x,y,W,b) 
    print dL_dW
    assert np.allclose(dL_dW, dL_dW_true, atol = 1e-3)



#-------------------------------------------------------------------------
def test_compute_dL_db():
    dL_dz = np.array([0.5, -0.5])
    dz_db = np.array([1.,1.])

    dL_db = compute_dL_db(dL_dz, dz_db) 
    
    assert type(dL_db) == np.ndarray 
    assert dL_db.shape == (2,) 

    dL_db_true = [0.5, -0.5]
    assert np.allclose(dL_db, dL_db_true, atol = 1e-3)


#-------------------------------------------------------------------------
def test_check_dL_db():
    p = np.random.randint(2,10) # number of features
    c = np.random.randint(2,10) # number of classes
    x = np.random.random(p)
    y = np.random.randint(c) 
    W = np.random.random((c,p))
    b = np.random.random(c)
    z, a, L = forward(x,y,W,b)
    dL_da, da_dz, dz_dW, dz_db = backward(x,y,a)
    dL_dz = compute_dL_dz(dL_da, da_dz)

    # analytical gradients
    dL_db = compute_dL_db(dL_dz,dz_db)
    # numerical gradients
    dL_db_true = check_dL_db(x,y,W,b) 
    
    assert np.allclose(dL_db, dL_db_true, atol = 1e-3)


#-------------------------------------------------------------------------
def test_update_W():

    W = np.array([[0., 0.,0.],[0.,0.,0.]])
    dL_dW = np.array([[ 0.5,  1.0,  1.5],
                      [-0.5, -1.0, -1.5]])

    W = update_W(W, dL_dW, alpha=1.) 

    W_true = [[-0.5,-1.,-1.5],
               [ 0.5, 1., 1.5]]
    assert np.allclose(W, W_true, atol = 1e-3)


    W = np.array([[0., 0.,0.],[0.,0.,0.]])
    W = update_W(W, dL_dW, alpha=10.)
    W_true = [[-5., -10., -15.],
               [ 5.,  10.,  15.]]
    assert np.allclose(W, W_true, atol = 1e-3)


#-------------------------------------------------------------------------
def test_update_b():
    b = np.array([0.,0.])
    dL_db = np.array([0.5, -0.5])

    b = update_b(b, dL_db, alpha=1.)

    b_true = [-0.5,0.5]
    assert np.allclose(b, b_true, atol = 1e-3)

    b = np.array([0.,0.])
    b = update_b(b, dL_db, alpha=10.)
    b_true = [-5.,5.]
    assert np.allclose(b, b_true, atol = 1e-3)


#-------------------------------------------------------------------------
def test_train():
    ''' test the correctness of train() function'''
    # set random seed to be 0
    np.random.seed(0)
    # an example feature matrix (4 instances, 2 features)
    Xtrain  = np.array( [[0., 1.],
                         [1., 0.],
                         [0., 0.],
                         [1., 1.]])
    Ytrain = np.array([0, 1, 0, 1])

    # call the function
    W, b = train(Xtrain, Ytrain)
   
    assert b[0] > b[1] # x3 is negative 
    assert W[1][0] + W[1][1] + b[1] > W[0][0] + W[0][1] + b[0] # x4 is positive
    assert W[1][1] + b[1] < W[0][1] + b[0] # x1 is negative 
    assert W[1][0] + b[1] > W[0][0] + b[0] # x2 is positive 

  
#-------------------------------------------------------------------------
def test_predict():
    ''' test the correctness of predict() function'''

    # an example feature matrix (4 instances, 2 features)
    Xtest  = np.array( [ [0., 1.],
                         [1., 0.],
                         [0., 0.],
                         [1., 1.]])
    
    W = np.array([[0.4, 0.],
                  [0.6, 0.]])
    b = np.array([0.1, 0.])

    # call the function
    Ytest, Ptest = predict(Xtest, W, b )
    print 'Ytest, Ptest:',Ytest, Ptest

    assert type(Ytest) == np.ndarray 
    assert Ytest.shape == (4,)
    assert type(Ptest) == np.ndarray 
    assert Ptest.shape == (4,2)

    Ytest_true = [0, 1,0, 1]
    Ptest_true = [[ 0.52497919, 0.47502081],
                  [ 0.47502081, 0.52497919],
                  [ 0.52497919, 0.47502081],
                  [ 0.47502081, 0.52497919]] 

    # check the correctness of the result 
    assert np.allclose(Ytest, Ytest_true, atol = 1e-2)
    assert np.allclose(Ptest, Ptest_true, atol = 1e-2)


#-------------------------------------------------------------------------
def test_softmax_regression():
    ''' test the correctness of both train() and predict() function'''
    # create a multi-class classification dataset
    n_samples = 200
    X,y = make_classification(n_samples= n_samples,
                              n_features=5, n_redundant=0, n_informative=4,
                              n_classes= 3,
                              class_sep = 4.,
                              random_state=1)
        
    Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
    w,b = train(Xtrain, Ytrain,alpha=.01, n_epoch=100)
    Y, P = predict(Xtrain, w, b)
    accuracy = sum(Y == Ytrain)/(n_samples/2.)
    print 'Training accuracy:', accuracy
    assert accuracy > 0.9
    Y, P = predict(Xtest, w, b)
    accuracy = sum(Y == Ytest)/(n_samples/2.)
    print 'Test accuracy:', accuracy
    assert accuracy > 0.9

