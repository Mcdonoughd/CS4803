from problem1 import *
import numpy as np

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_D():
    ''' test the correctness of compute_D() function in problem3.py'''

    #-------------------------------
    # an example adjacency matrix (3 nodes)
    A = np.array([[0., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 0.]])

    # call the function
    D = compute_D(A)

    # true answer
    D_true = np.array([[1., 0., 0.],
                       [0., 2., 0.],
                       [0., 0., 1.]])

    # test the result
    assert np.allclose(D,D_true)



#-------------------------------------------------------------------------
def test_compute_L():
    ''' test the correctness of compute_L() function in problem3.py'''

    #-------------------------------
    # an example adjacency matrix (3 nodes)
    A = np.array([[0., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 0.]])

    # call the function
    L = compute_L(A)

    # true answer
    L_true = np.array([[ 1.,-1., 0.],
                       [-1., 2.,-1.],
                       [ 0.,-1., 1.]])

    # test the result
    assert np.allclose(L,L_true)




#-------------------------------------------------------------------------
def test_spectral_clustering():
    ''' test the correctness of spectral_clustering() function in problem3.py'''

    #-------------------------------
    # an example adjacency matrix (2 groups without any link between the two groups) 
    A = np.array([[0., 1., 0., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., 1., 0.]])
    # make sure matrix A is symmetric
    assert np.allclose(A, A.T)

    # call the function
    x = spectral_clustering(A)

    # test the correctness of the result
    assert np.allclose([0,0,1,1],x) or np.allclose([1,1,0,0],x)

    #-------------------------------
    # an example adjacency matrix (2 groups with a link between the two groups) 
    A = np.array([[0., 1., 1., 0., 0., 0.],
                  [1., 0., 1., 0., 0., 0.],
                  [1., 1., 0., 1., 0., 0.],
                  [0., 0., 1., 0., 1., 1.],
                  [0., 0., 0., 1., 0., 1.],
                  [0., 0., 0., 1., 1., 0.]])
    # make sure matrix A is symmetric
    assert np.allclose(A, A.T)

    # call the function
    x = spectral_clustering(A)

    # test the correctness of the result
    assert np.allclose([0,0,0,1,1,1],x) or np.allclose([1,1,1,0,0,0],x)


