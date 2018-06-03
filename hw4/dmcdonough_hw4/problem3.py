import numpy as np
from problem2 import PCA
from sklearn.datasets import fetch_olivetti_faces

# -------------------------------------------------------------------------
'''
    Problem 3: eigen faces 
    In this problem, you will use PCA to compute eigen faces in a face image dataset.
    You need to install the following package:
        * sklearn
        * scipy
        * pillow
    You could use `pip install sklearn` to install the package.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''


# --------------------------
def load_dataset():
    '''
        Load (or download if not exist) the olivetti face image dataset (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
        Output:
            X:  the feature matrix, a float numpy matrix of shape (400, 4096). Here 400 is the number of images, 4096 is the number of features.
            l:  labels associated to each face image. Those labels are ranging from 0-39 and correspond to the Subject IDs. 
            images: numpy array of shape (400, 64, 64). Each face image is a (64, 64) matrix, and we have 400 images in the dataset.
    '''

    # download (or load local) dataset
    dataset = fetch_olivetti_faces()

    # the images
    X = dataset.data

    # the label to predict is the id of the person
    l = dataset.target
    images = dataset.images

    # statistics of the data
    n_samples, h, w = images.shape

    n_features = X.shape[1]
    print("Total dataset size:")
    print("n_images: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("image height: %d" % h)
    print("image width: %d" % w)
    return X, l, images


# --------------------------
def centering_X(X):
    '''
        Given a face image dataset, compute centered matrix X. This function may take 1-5 minutes to run.
        Input:
            X:  the feature matrix, a float numpy matrix of shape (400, 4096). Here 400 is the number of images, 4096 is the number of features.
        Output:
            Xc:  the centered feature matrix, a float numpy matrix of shape (400, 4096). 
            mu:  the average face image, a float numpy matrix of shape (64,64). 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    mu = X.mean(axis=0)
    # print mu.shape
    # print mu
    X -= mu
    mu.shape = (64, 64)
    Xc = X

    #########################################
    return Xc, mu


# --------------------------
def olivetti_eigen_faces(K=20):
    '''
        Compute top K eigen faces of the olivetti face image dataset.
        Input:
            K:  the number of eigen face to keep. 
        Output:
            W:  the eigen faces, a float numpy array of shape (K,64,64). 
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape (400, K). 
        Note: this function may take 1-5 minutes to run, and 1-2GB of memory while running.
    '''

    #########################################
    ## INSERT YOUR CODE HERE

    X, _, _ = load_dataset()
    Xc, _ = centering_X(X)

    Xp, W = PCA(Xc, K)
    # W.shape = (64, 64, 20)
    #print W
    W = W.swapaxes(0,1)
    #print W

    W.shape = (20, 64, 64)
    #W = W.T
    #print Xp
    #W.shape = (20, 64, 64)

    #########################################
    return W, Xp
