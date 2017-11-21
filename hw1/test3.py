from problem3 import *

'''
    Unit test 3: 
    This file includes unit tests for problem3.py. 
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''


#-------------------------------------------------------------------------
def test_compute_P():
    ''' test the correctness of compute_P() function in problem3.py'''

    # adjacency matrix of shape (3 by 3)
    A = np.array( [ [0., 1., 1.],
                    [1., 0., 0.],
                    [1., 1., 0.]])

    # call the function
    P= compute_P(A) 

    # test whether or not P is a numpy array
    assert type(P) == np.ndarray 

    # test the shape of the matrix
    assert P.shape == (3,3)
    
    # check the correctness of the result 
    P_real=np.array([[ 0. ,  0.5,  1. ],
                     [ 0.5,  0. ,  0. ],
                     [ 0.5,  0.5,  0. ]] )
    assert np.allclose(P, P_real)
   
    #---------------------
    # test with another matrix

    # test on another adjacency matrix of shape (2 by 2)
    A = np.array( [ [0., 1.],
                    [1., 0.]])
    
    # call the function again
    P = compute_P(A) 

    # test the shape of the matrix
    assert P.shape == (2,2)

    # check the correctness of the result 
    assert np.allclose(P, A)

#-------------------------------------------------------------------------
def test_random_walk_one_step():
    ''' test the correctness of random_walk_one_step() function in problem3.py'''

    # transition matrix of shape (3 by 3) 
    P = np.array([[ 0. ,  0.5,  1. ],
                  [ 0.5,  0. ,  0. ],
                  [ 0.5,  0.5,  0. ]] )

    # an all-one vector of shape (3 by 1)
    x_i =  np.ones((3,1)) 
    
    # call the function 
    x_i_plus_1 = random_walk_one_step(P, x_i) 

    # test whether or not x_i_plus_1 is a numpy array
    assert type(x_i_plus_1) == np.ndarray 

    # check the shape of the vector
    assert x_i_plus_1.shape == (3,1)

    # check the correctness of the result 
    x_real=np.array([[1.5],
                     [0.5],
                     [1.0]] )
    assert np.allclose(x_real, x_i_plus_1)

    #---------------------
    # test with another matrix

    # another transition matrix of shape (2 by 2) 
    P = np.array([[ 0.1,  0.4],
                  [ 0.9,  0.6]])

    # an all-one vector of shape (2 by 1)
    x_i =  np.ones((2,1)) 

    # call the function 
    x_i_plus_1 = random_walk_one_step(P, x_i) 

    # check the shape of the vector
    assert x_i_plus_1.shape == (2,1)

    # check the correctness of the result 
    x_real=np.array([[0.5],
                     [1.5]] )
    assert np.allclose(x_real, x_i_plus_1)
 


#-------------------------------------------------------------------------
def test_random_walk():
    ''' test the correctness of random_walk() function in problem3.py'''

    # a transition matrix of shape (3 by 3) 
    P = np.array([[ 0. ,  0.5,  1. ],
                  [ 0.5,  0. ,  0. ],
                  [ 0.5,  0.5,  0. ]] )
 
    # an all-one vector of shape (3 by 1)
    x_0 =  np.ones((3,1)) 

    # call the function 
    x, n_steps = random_walk(P, x_0) 

    # check the shape of the vector
    assert x.shape == (3,1)

    # check number of random walks 
    assert n_steps == 18

    # check the correctness of the result 
    x_real = np.array( [[ 1.33333333],
                        [ 0.66666667],
                        [ 1.        ]] )
    assert np.allclose(x_real, x)
    
    #---------------------
    # test with another matrix

    # another transition matrix of shape (2 by 2) 
    P = np.array([[ 0.5,  0.5],
                  [ 0.5,  0.5]])

    # an all-one vector of shape (2 by 1)
    x_0 =  np.ones((2,1)) 

    # call the function 
    x, n_steps = random_walk(P, x_0) 

    # test whether or not x is a numpy array
    assert type(x) == np.ndarray 

    # check the shape of the vector
    assert x.shape == (2,1)

    # check number of random walks 
    assert n_steps == 1
    print n_steps

    # check the correctness of the result 
    x_real=np.array([[1.],
                     [1.]] )
    assert np.allclose(x_real, x)


#-------------------------------------------------------------------------
def test_pagerank_v1():
    ''' test the correctness of pagerank_v1() function in problem3.py'''

    # adjacency matrix of shape (3 by 3)
    A = np.array( [ [0., 1., 1.],
                    [1., 0., 0.],
                    [1., 1., 0.]])
    
    # call the function
    x= pagerank_v1(A) 

    # test whether or not x is a numpy array
    assert type(x) == np.ndarray 

    # test the shape of the vector
    assert x.shape == (3,1)

    # check the correctness of the result 
    x_real = np.array( [[ 1.33333333],
                        [ 0.66666667],
                        [ 1.        ]] )
    assert np.allclose(x_real, x)

