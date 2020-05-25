from problem4 import *

'''
    Unit test 4: 
    This file includes unit tests for problem4.py. 
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_S():
    ''' test the correctness of compute_S() function in problem4.py'''

    # adjacency matrix of shape (3 by 3)
    A = np.array( [ [0., 1., 0.],
                    [1., 0., 0.],
                    [1., 1., 0.]])

    # call the function
    S = compute_S(A) 

    # test whether or not S is a numpy array
    assert type(S) == np.ndarray 

    # test the shape of the matrix 
    assert S.shape == (3,3)
    
    # check the correctness of the result 
    S_real=np.array([[ 0. ,  0.5,  0.333333 ],
                     [ 0.5,  0. ,  0.333333 ],
                     [ 0.5,  0.5,  0.333333 ]] )
    assert np.allclose(S, S_real)
   
    #---------------------
    # test with another matrix (no sink node)

    # test on another adjacency matrix of shape (2 by 2)
    A = np.array( [ [0., 1.],
                    [1., 0.]])
    
    # call the function again
    S = compute_S(A) 

    # test the shape of the matrix 
    assert S.shape == (2,2)

    # check the correctness of the result 
    assert np.allclose(S, A)

    #---------------------
    # test with a sink node 

    # test on another adjacency matrix of shape (2 by 2)
    A = np.array( [ [0., 0.],
                    [1., 0.]])

    # call the function again
    S = compute_S(A) 

    # test the shape of the matrix 
    assert S.shape == (2,2)

    # check the correctness of the result 
    S_real = np.array( [ [0., 0.5],
                         [1., 0.5]])
    assert np.allclose(S, S_real)

#-------------------------------------------------------------------------
def test_pagerank_v2():
    ''' test the correctness of pagerank_v2() function.'''

    # adjacency matrix of shape (3 by 3)
    A = np.array( [ [0., 1., 1.],
                    [1., 0., 0.],
                    [1., 1., 0.]])
    
    # call the function
    x= pagerank_v2(A) 

    # test the shape of the vector
    assert x.shape == (3,1)

    # check the correctness of the result 
    x_real = np.array( [[ 1.33333333],
                        [ 0.66666667],
                        [ 1.        ]] )
    assert np.allclose(x_real, x)

