from problem2 import *

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''


#-------------------------------------------------------------------------
def test_matrix_vector_multiplication():
    ''' test the correctness of matrix_vector_multiplication() function in problem2.py'''

    # create a matrix  [[1., 2.],
    #                   [3., 4.],
    #                   [5., 6.]]
    # of shape (3 by 2)
    X = np.array([[1.,2.],[3.,4.],[5.,6.]])

    # create a vector of shape (2 by 1): [[1.],
    #                                     [2.]]
    y = np.array([[1.],[2.]])

    # call the function
    z= matrix_vector_multiplication(X, y)

    # test whether or not z is a numpy array
    assert type(z) == np.ndarray 

    # test the shape of the vector
    assert z.shape == (3,1)

    # check the correctness of the result 
    z_real = np.reshape(np.array([5., 11., 17.]),(3,1))
    assert np.allclose(z, z_real)
