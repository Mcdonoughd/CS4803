from problem2 import *
import numpy as np

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_update_U():
    ''' test the correctness of update_U() function in problem2.py'''

    #-------------------------------
    # an example rating matrix (2 movies, 2 users)
    R = np.array([[2., 2.],
                  [2., 2.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.] ]) # k=1

    # call the function
    U_new = update_U(R,V, U, beta=1., mu = 1.)

    # true answer
    U_true = np.array([[3.],
                       [3.] ]) # k=1

    # test the result
    assert np.allclose(U_new,U_true)

    #-------------------------------
    # an example rating matrix (3 movies, 2 users)
    R = np.array([[2., 2.],
                  [2., 2.],
                  [2., 2.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.],
                  [1.] ]) # k=1

    # call the function
    U_new = update_U(R,V, U, beta=1., mu=1.)

    # true answer
    U_true = np.array([[3.],
                       [3.],
                       [3.] ]) # k=1

    # test the result

    assert np.allclose(U_new,U_true)

    #-------------------------------
    # an example rating matrix (2 movies, 2 users)
    R = np.array([[1., 2.],
                  [3., 4.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.] ]) # k=1

    # call the function
    U_new = update_U(R,V, U, beta=1., mu = 1.)

    # true answer
    U_true = np.array([[1.],
                       [9.] ]) # k=1
    # test the result
    assert np.allclose(U_new,U_true)

    # call the function
    U_new = update_U(R,V, U, beta=2., mu = 1.)

    # true answer
    U_true = np.array([[1.],
                       [17.] ]) # k=1
    # test the result
    assert np.allclose(U_new,U_true)

    # call the function
    U_new = update_U(R,V, U, beta=1., mu = 0.)

    # true answer
    U_true = np.array([[3.],
                       [11.] ]) # k=1
    # test the result
    assert np.allclose(U_new,U_true)
    
    #-------------------------------
    # an example rating matrix (2 movies, 2 users) with missing ratings
    R = np.array([[2., 0.],
                  [0., 2.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.] ]) # k=1

    # call the function
    U_new = update_U(R,V, U, beta=1., mu = 1.)

    # test the result
    assert np.allclose(U_new,U)

    #-------------------------------
    # an example rating matrix (2 movies, 2 users) when K = 2
    R = np.array([[2., 2.],
                  [2., 2.]])

    V = np.array([[1., 1.],
                  [1., 1.] ]) # k=2
    U = np.array([[1., 1.],
                  [1., 1.] ]) # k=2

    # call the function
    U_new = update_U(R,V, U, beta=1., mu = 1.)

    # test the result
    assert np.allclose(U_new,-U)


#-------------------------------------------------------------------------
def test_update_V():
    ''' test the correctness of update_V() function in problem2.py'''

    #-------------------------------
    # an example rating matrix (2 movies, 2 users)
    R = np.array([[2., 2.],
                  [2., 2.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.] ]) # k=1

    # call the function
    V_new = update_V(R,U, V, beta=1., mu = 1.)

    # true answer
    V_true = np.array([[3., 3.]]) # k=1

    # test the result
    assert np.allclose(V_new,V_true, atol= 1e-1)

    #-------------------------------
    # an example rating matrix (3 movies, 2 users)
    R = np.array([[2., 2.],
                  [2., 2.],
                  [2., 2.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.],
                  [1.] ]) # k=1

    # call the function
    V_new = update_V(R,U, V, beta=1., mu=1.)

    # true answer
    V_true = np.array([[5., 5.]]) # k=1

    # test the result
    assert np.allclose(V_new,V_true)


    #-------------------------------
    # an example rating matrix (2 movies, 2 users)
    R = np.array([[1., 2.],
                  [3., 4.]])

    V = np.array([[1., 1.]]) # k=1
    U = np.array([[1.],
                  [1.] ]) # k=1

    # call the function
    V_new = update_V(R, U, V, beta=1., mu = 1.)

    # true answer
    V_true = np.array([[3., 7.]]) # k=1

    # test the result
    assert np.allclose(V_new,V_true)

    # call the function
    V_new = update_V(R,U, V, beta=2., mu = 1.)

    # true answer
    V_true = np.array([[5., 13.]]) # k=1

    # test the result
    assert np.allclose(V_new,V_true)

    # call the function
    V_new = update_V(R, U, V, beta=1., mu = 0.)

    # true answer
    V_true = np.array([[5., 9.]]) # k=1

    # test the result
    assert np.allclose(V_new, V_true)

    #-------------------------------
    # an example rating matrix (2 movies, 2 users) when K = 2
    R = np.array([[2., 2.],
                  [2., 2.]])

    V = np.array([[1., 1.],
                  [1., 1.] ]) # k=2
    U = np.array([[1., 1.],
                  [1., 1.] ]) # k=2

    # call the function
    V_new = update_V(R,U, V, beta=1., mu = 1.)

    # test the result
    assert np.allclose(V_new,-V)



#-------------------------------------------------------------------------
def test_matrix_decoposition():
    ''' test the correctness of matrix_decoposition() function in problem2.py'''

    #-------------------------------
    # an example rating matrix (2 movies, 2 users)
    R = np.array([[1., 1.],
                  [1., 1.]])
    # call the function
    U, V = matrix_decoposition(R,1)

    # test whether or not the result is a float number 
    assert type(U) == np.ndarray 
    assert type(V) == np.ndarray 
    assert U.shape == (2,1)
    assert V.shape == (1,2)

    # check the correctness of the result 
    assert np.allclose(np.dot(U,V),R, atol=0.1)


    #-------------------------
    # another example
    
    # a random rating matrix
    R = np.random.randint(1,6, (10, 5)).astype(float)

    # call the function
    U, V = matrix_decoposition(R,5)

    # test whether or not the result is a float number 
    assert type(U) == np.ndarray 
    assert type(V) == np.ndarray 
    assert U.shape == (10,5)
    assert V.shape == (5,5)

    # check the correctness of the result 
    assert np.allclose(np.dot(U,V), R, atol=.1)
