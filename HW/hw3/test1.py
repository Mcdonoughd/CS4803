from problem1 import *
import numpy as np

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_cosine_similarity():
    ''' test the correctness of cosine_similarity() function in problem1.py'''

    # an example ratings of two users (without missing values)
    RA = [ 5., 3.]
    RB = [ 1., 4.]

    # call the function
    S = cosine_similarity(RA, RB)

    # test whether or not the result is a float number 
    assert type(S) == float 

    # check the correctness of the result 
    assert np.allclose(S, 0.70710678)

    #-------------------------
    # another example with missing values 
    RA = [0. , 0., 5., 3., 0., 2., 0. ]
    RB = [0.3, 0., 1., 4., 0., 0., 0. ]

    # call the function
    S = cosine_similarity(RA, RB)

    # test whether or not the result is a float number 
    assert type(S) == float 

    # check the correctness of the result 
    assert np.allclose(S, 0.70710678)
    #assert np.allclose(S, 0)
    #-------------------------
    # test another example with non-overlapping vectors

    # an example ratings of two users
    RA = [0. , 0., 0., 0., 0., 2., 0. ]
    RB = [0.3, 0., 1., 4., 0., 0., 0. ]

    # call the function
    S = cosine_similarity(RA, RB)

    # check the correctness of the result 
    assert np.allclose(S, 0.)


#-------------------------------------------------------------------------
def test_find_users():
    ''' test the correctness of find_users() function in problem1.py'''

    # an example rating matrix
    R = np.array( [ [0., 1., 3.],
                    [2., 0., 0.],
                    [4., 5., 0.]])
    
    # call the function
    idx = find_users(R, 0) 

    # test whether or not the result is a python list 
    assert type(idx) == list 
    assert type(idx[0]) == int
    
    # check the correctness of the result 
    assert idx == [1,2]

    #-------------------------
    # test another example
    idx = find_users(R, 2) 

    # check the correctness of the result 
    assert idx == [0,1]



#-------------------------------------------------------------------------
def test_user_similarity():
    ''' test the correctness of user_similarity() function in problem1.py'''

    # an example rating matrix
    R = np.array( [ [0., 1., 3., 1.],
                    [1., 0., 0., 0.],
                    [1., 1., 0., 1.]])
    
    # call the function (compute the similarity between user 0 and 3, and between user 1 and 3.
    sim = user_similarity(R, 3, [0,1]) 

    # test whether or not the result is a python list 
    assert type(sim) == list 
    assert type(sim[0]) == float
    
    # check the correctness of the result 
    assert np.allclose(sim, [1.,1.])

    #-------------------------
    # test another example
    R = np.array( [ [1., 1., 2., 1.],
                    [1., 0., 0., 0.],
                    [1., 1., 1., 2.]])

    # check the correctness of the result 
    sim = user_similarity(R, 2, [2,3])

    assert np.allclose(sim, [1.,.8])


#-------------------------------------------------------------------------
def test_user_based_prediction():
    ''' test the correctness of user_based_prediction() function in problem1.py'''

    # an example rating matrix
    R = np.array( [ [0., 1., 3., 2., 1.],
                    [1., 0., 0., 3., 2.],
                    [1., 1., 1., 5., 1.]])

    # call the function

    p = user_based_prediction(R, 1, 2, K=2)

    # test the correctness of the result
    assert np.allclose(p, 1.472135955)

    # call the function 
    p = user_based_prediction(R, 1, 1, K=2)

    # test the correctness of the result
    assert np.allclose(p, 1.5)


    # call the function (K is larger than the number of available ratings, use all) 
    p = user_based_prediction(R, 1, 2, K=5)

    # test the correctness of the result
    assert np.allclose(p, 1.86062745284)

    # an example rating matrix where there is no rating for 2nd movie
    R = np.array( [ [0., 1., 3., 2., 1.],
                    [0., 0., 0., 0., 0.],
                    [1., 1., 1., 5., 1.]])

    # call the function (K is larger than the number of available ratings, use all) 
    p = user_based_prediction(R, 1, 2, K=5)

    # test the correctness of the result (default rating is 3
    assert p == 3.0

#-------------------------------------------------------------------------
def test_compute_RMSE():
    ''' test the correctness of compute_RMSE() function in problem1.py'''
    
    # an example 
    ratings_pred = [1., 2., 3.]    
    ratings_real = [1., 2., 3.]    

    # call the function 
    RMSE = compute_RMSE(ratings_pred,ratings_real)

    # test whether or not the result is a python list 
    assert type(RMSE) == float

    # test the correctness of the result
    assert RMSE == 0.

    # another example 
    ratings_pred = [3., 1., 2.]    
    ratings_real = [4., 2., 3.]    

    # call the function 
    RMSE = compute_RMSE(ratings_pred, ratings_real)

    # test the correctness of the result
    assert RMSE == 1.

    # another example 
    ratings_pred = [1., 1., 2.]    
    ratings_real = [3., 2., 3.]    

    # call the function 
    RMSE = compute_RMSE(ratings_pred, ratings_real)

    # test the correctness of the result
    assert np.allclose(RMSE, 1.41421356237)



#-------------------------------------------------------------------------
def test_load_rating_matrix():
    ''' test the correctness of load_rating_matrix() function in problem1.py'''
    # call the function 
    R = load_rating_matrix('movielens_train.csv')

    assert type(R) == np.ndarray
    assert R.shape ==(1682, 943)
    assert R[0,0] ==5.  
    assert R[0,1] ==4.  
    assert R[1,-1] ==5.  
    assert R[-1,-1] ==0.  


#-------------------------------------------------------------------------
def test_load_test_data():
    ''' test the correctness of load_test_data() function in problem1.py'''
    # call the function 
    m_ids, u_ids, ratings = load_test_data('movielens_test.csv')

    assert type(m_ids) == list
    assert type(m_ids[0]) == int
    assert len(m_ids) == 1000

    assert type(u_ids) == list
    assert type(u_ids[0]) == int
    assert len(u_ids) == 1000

    assert type(ratings) == list
    assert type(ratings[0]) == float
    assert len(ratings) == 1000

    assert u_ids[:3] ==[0, 0, 0] 
    assert m_ids[:3] ==[5, 9, 11] 
    assert ratings[:3] ==[5., 3., 5.] 


#-------------------------------------------------------------------------
def test_movielens_user_based():
    ''' test the correctness of movielens_user_based() function in problem1.py'''
    
    # call the function 
    RMSE = movielens_user_based()
    assert np.allclose(RMSE, 1.11871095933)













