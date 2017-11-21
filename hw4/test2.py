from problem2 import *
import numpy as np

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_C():
    ''' test the correctness of compute_C() function in problem2.py'''

    #-------------------------------
    # an example matrix (2 dimensional)
    X = np.array([[0., 2.],
                  [2., 0.],
                  [1., 1.]])

    # call the function
    C = compute_C(X)

    # true answer
    C_true = np.array([[1, -1],
                       [-1, 1]])

    # test the result
    assert np.allclose(C,C_true, atol=1e-3)


    #-------------------------------
    # an example matrix (3 dimensional)
    X = np.array([[0., 2., 2.],
                  [2., 0., 0]])

    # call the function
    C = compute_C(X)

    # true answer
    C_true = np.array([[ 2, -2,-2],
                       [-2,  2, 2],
                       [-2,  2, 2]])

    # test the result
    assert np.allclose(C,C_true)

#-------------------------------------------------------------------------
def test_PCA():
    ''' test the correctness of PCA() function in problem2.py'''

    #-------------------------------
    # an example matrix 
    #X = np.random.random((100,10)) # generate an N = 100, D = 10 random data matrix
    X = np.array([[0., 2.],
                  [2., 0.],
                  [1., 1.]])

    # call the function
    Xp, P = PCA(X)
 
    assert Xp.shape ==(3,1)
    assert P.shape ==(2,1)
  

    P_true = np.array([[ .707],
                       [-.707]])
    Xp_true = np.array([[-1.414],
                        [ 1.414],
                        [ 0    ]])
 
    # test the result
    assert np.allclose(P,P_true, atol=1e-3) or np.allclose(P,-P_true, atol=1e-3)
    assert np.allclose(Xp, Xp_true, atol=1e-3) or np.allclose(Xp, -Xp_true, atol=1e-3)

    #-------------------------------
    # an example matrix 
    X = np.array([[0., 2., 2.],
                  [2., 0., 0]])

    # call the function
    Xp, P = PCA(X)

    assert Xp.shape ==(2,1)
    assert P.shape ==(3,1)

    # test the result
    P_true = np.array([[ .577],
                       [-.577],
                       [-.577]])
    Xp_true = np.array([[-2.3094],
                        [ 1.1547]])

    assert np.allclose(P,P_true, atol=1e-3) or np.allclose(P,-P_true, atol=1e-3)
    assert np.allclose(Xp,Xp_true, atol=1e-3) or np.allclose(Xp,-Xp_true, atol=1e-3)

    #-------------------------------
    # an example matrix 
    #X = np.random.random((100,10)) # generate an N = 100, D = 10 random data matrix
    X = np.array([[2., 2.],
                  [0., 0]])


    # call the function
    Xp, P = PCA(X,2)

    assert Xp.shape ==(2,2)
    assert P.shape ==(2,2)

    P_true = np.array([[.707, -.707],
                       [.707,  .707]])

    Xp_true = np.array([[2.828, 0],
                        [0    , 0]])
    assert np.allclose(P,P_true, atol=1e-3) or np.allclose(P,-P_true, atol=1e-3)
    assert np.allclose(Xp,Xp_true, atol=1e-3) or np.allclose(Xp,-Xp_true, atol=1e-3)



