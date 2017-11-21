import numpy as np
import math

# -------------------------------------------------------------------------
'''
    Problem 2: PCA 
    In this problem, you will implement a version of the principal component analysis method to reduce the dimensionality of data.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''


# --------------------------
def compute_C(X):
    '''
        Compute the covariance matrix C. 
        Input:
            X:  the feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
        Output:
            C:  the covariance matrix, a numpy float matrix of shape p by p. 
    '''

    #########################################
    ## INSERT YOUR CODE HERE
    C = np.cov(X.T, ddof=0)
    #print C
    #########################################
    return C


# --------------------------
def PCA(X, d=1):
    '''
        Compute PCA of matrix X. 
        Input:
            X:  the feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
            d:  the number of dimensions to output (d should be smaller than p)
        Output:
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape n by d. 
             P: the projection matrix, a numpy float matrix of shape p by d. 
        Hint: you could use np.linalg.eig() to compute the eigen vectors of a matrix. 
        Note: in this problem, you cannot use existing package for PCA, such as scikit-learn
    '''

    #########################################
    ## INSERT YOUR CODE HERE
    # X = n p
    # W = p d
    # Y = n d

    C = compute_C(X)
    #print C
    eig_vals, eig_vecs = np.linalg.eig(C)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    #print X.shape[1]
    matrix_w = (np.zeros(shape=(X.shape[1],d)))
    #print matrix_w
    for i in range(d):
        for j in range(X.shape[1]):
            matrix_w[j][i] = eig_pairs[i][1][j]
    #print matrix_w
    P = matrix_w
    Xp = np.dot(X,P)
    #print Xp
    #########################################
    return Xp, P
