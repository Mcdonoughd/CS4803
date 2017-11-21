from problem3 import * 
import numpy as np
import scipy.misc

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_load_dataset():
    ''' test the correctness of load_dataset() function in problem3.py'''

    # call the function
    X, l, images = load_dataset()
    assert X.shape == (400,4096)
    assert l.shape == (400,)
    assert images.shape == (400,64,64)



#-------------------------------------------------------------------------
def test_centering_X():
    ''' test the correctness of centering_X() function in problem3.py'''

    # call the function
    X, _, _ = load_dataset()
    Xc, c = centering_X(X)
    
    assert type(c) == np.ndarray
    assert type(Xc) == np.ndarray

    assert Xc.shape == (400,4096)
    assert c.shape == (64,64)
    assert sum(Xc.mean(axis=0)) < 1e-3

    # write average face image to file 'mu.jpg'
    scipy.misc.imsave('mu.jpg', c)
    
    assert np.allclose([0.40013435, 0.43423545, 0.4762809],c[0,:3],atol=1e-2)
    assert np.allclose([0.36046496, 0.36678693, 0.37106389],c[-1,:3],atol=1e-2)

#-------------------------------------------------------------------------
def test_olivetti_eigen_faces():
    ''' test the correctness of olivetti_eigen_faces() function in problem3.py'''

    # call the function
    W,Xp = olivetti_eigen_faces()
   
    assert W.shape == (20,64,64)
    assert Xp.shape == (400,20)
    for i in range(20):
        x = W[i]
        scipy.misc.imsave('eigen_face_%d.jpg' % (i+1), x) 

    c = W[0]
    assert np.allclose([-0.0041911,  -0.0071095,  -0.00933609],c[0,:3],atol=1e-3)
    assert np.allclose([0.01161627,  0.01290446,  0.01308161],c[-1,:3],atol=1e-3)

    # save the results into a file
    np.save('face_pca.npy',Xp)
