from problem3 import load_dataset 
from problem4 import * 
import numpy as np
import scipy.misc

'''
    Unit test 4:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_compute_distance():
    ''' test the correctness of compute_distance() function in problem4.py'''
    #-------------------------------
    # an example matrix (two features, 3 instances)
    X = np.array([[0., 1.],
                  [1., 0.],
                  [0., 0.]])
    q = np.array([1., 1.])
    # call the function
    s = compute_distance(X,q)
    
    assert type(s) == np.ndarray
    assert s.shape == (3,)
    assert s.dtype == float

    assert np.allclose(s, [1,1,1.41421356])





#-------------------------------------------------------------------------
def test_face_recogition():
    ''' test the correctness of face_recogition() function in problem4.py'''
    #-------------------------------
    # an example matrix (two features, 3 instances)
    X = np.array([[0., 1.],
                  [1., .5],
                  [0., 0.]])
    q = np.array([1., 1.])
    # call the function
    ids = face_recogition(X,q)
    
    assert type(ids) == np.ndarray
    assert ids.shape == (3,)
    assert ids.dtype == int 

    assert ids.tolist() == [1,0,2]

#-------------------------------------------------------------------------
def test_face_recogition_olivetti():
    ''' test the correctness of face_recogition_olivetti() function in problem4.py'''
    #-------------------------------
    qid = 70  # use the first image as the query image

    # call the function
    ids =face_recogition_olivetti(qid)
    
    assert type(ids) == np.ndarray
    assert ids.shape == (400,)
    assert ids.dtype == int 

    # the most similar image should be the query image itself
    #assert ids[0] == qid
    X, l, images = load_dataset()
    scipy.misc.imsave('query.jpg', images[qid]) 

    # save the top 10 similar images to the query
    for i in range(10):
        x = images[ids[i]]
        scipy.misc.imsave('result_%d.jpg' % (i+1), x) 
    
    print ids
    assert ids[:4].tolist() ==[70, 71, 75, 74]
