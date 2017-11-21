import numpy as np
import math

# -------------------------------------------------------------------------
'''
    Problem 3: face recognition 
    In this problem, you will use PCA to perform face recogition in a face image dataset.
    We assume you have already passed the unit tests in problem3.py. There will be a file named "face_pca.npy" in your folder.
    This file contains a numpy matrix of shape (400,20), which is the reduced dimsions for the 400 face images.
    We will need to use this file in this problem.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''


# --------------------------
def compute_distance(X, q):
    '''
        Compute the Euclidean distance between an query image and all the images in an image dataset.  
        Intput:
            X: the feature matrix, a float numpy matrix of shape (400, 20). Here 400 is the number of images,  20 is the number of features.
            q:  a query of face image. a numpy vector of shape (1, 20). 
        Output:
            d: distances between the query image and all the images in X. A numpy vector of shape (400,1), where each element i, is the Euclidean distance between i-th image in X and the query image.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    d = np.empty(shape=(X.shape[0],))
    for i in range(len(X)):
        d[i] = np.linalg.norm(X[i] - q)
    # print type(d)
    # print d


    #########################################
    return d


# --------------------------
def face_recogition(X, q):
    '''
        Compute the most similar faces to the query face image based upon descending order of Euclidean distances between the query image and all the images in an image dataset.  
        Intput:
            X: the feature matrix, a float numpy matrix of shape (400, 20). Here 400 is the number of images,  20 is the number of features.
            q:  a query of face image. a numpy vector of shape (1, 20). 
        Output:
            ids: the ranked indexes of face images (from the most similar face image to the least similar face image).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    d = compute_distance(X, q)
    ids = np.argsort(d)
    # print type(ids)

    print d
    print ids

    #########################################
    return ids


# --------------------------
def face_recogition_olivetti(qid):
    '''
        Compute the most similar faces to the query face (id) from all the images in an image dataset.  
        We will use one image from olivetti face dataset as the query and search for similar faces to the query.
        Intput:
            q:  a query of face image id. a scalar integer between 0 and 399 
        Output:
            ids: the ranked indexes of face images (from the most similar face image to the least similar face image).
    '''

    # load the PCA results from problem 3.
    X = np.load('face_pca.npy')
    #########################################
    ## INSERT YOUR CODE HERE
    ids = face_recogition(X, qid)
    # print ids


    #########################################
    return ids
