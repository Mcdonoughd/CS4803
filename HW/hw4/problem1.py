import numpy as np
import math

# -------------------------------------------------------------------------
'''
    Problem 1: Graph Clustering (Spectral Clustering) 
    In this problem, you will implement a version of the spectral clustering method to cluster the nodes in a graph into two groups.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''


# --------------------------
def compute_D(A):
    '''
        Compute the degree matrix D. 
        Input:
            A:  the adjacency matrix, a float numpy matrix of shape n by n. Here n is the number of nodes in the network.
                If there is a link between node i an node j, then A[i][j] = A[j][i] = 1. 
        Output:
            D:  the degree matrix, a numpy float matrix of shape n by n. 
                All off-diagonal elements are 0. Each diagonal element represents the degree of the node (number of links). 
    '''

    #########################################
    ## INSERT YOUR CODE HERE
    D = np.copy(A)
    D[D != 0] = 0
    for i in range(len(A)):
        num_links = 0
        for j in range(len(A[:, 0])):
            if A[j][i] != 0:
                num_links += 1
        D[i][i] = num_links
    # print D
    #########################################
    return D


# --------------------------
def compute_L(A):
    '''
        Compute the Laplacian matrix L. 
        Input:
            A:  the adjacency matrix, a float numpy matrix of shape n by n. Here n is the number of nodes in the network.
                If there is a link between node i an node j, then A[i][j] = A[j][i] = 1. 
        Output:
            L:  the Laplacian matrix, a numpy float matrix of shape n by n. 
    '''

    #########################################
    ## INSERT YOUR CODE HERE
    L = np.subtract(compute_D(A), A)

    #########################################
    return L


# --------------------------
def spectral_clustering(A):
    '''
        Spectral clustering of a graph. 
        Input:
            A:  the adjacency matrix, a float numpy matrix of shape n by n. Here n is the number of nodes in the network.
                If there is a link between node i an node j, then A[i][j] = A[j][i] = 1. 
        Output:
            x:  the binary vector of shape (n by 1), a numpy float vector of (0/1) values.
                It indicates a binary partition on the graph, such as [1.,1.,1., 0.,0.,0.].
        Hint: you could use np.linalg.eig() to compute the eigen vectors of a matrix. 
        Hint: x is related to the eigen vector of L with the second smallest eigen values. 
              For example, if the eigen vector is [0.2,-0.1, -0.2], the values larger than zero will be 1, so x=[1,0,0] in this example.
        Note: you cannot use any existing python package for spectral clustering, such as scikit-learn.
    '''

    #########################################
    ## INSERT YOUR CODE HERE
    val, vec = np.linalg.eigh(A)
    greatest_val = 0
    second_greatest = 0
    # print val
    # print vec
    for i in range(len(val)):
        if val[i] > val[greatest_val]:
            second_greatest = greatest_val
            greatest_val = i
        elif val[i] > val[second_greatest]:
            second_greatest = i
    x = vec[:, int(second_greatest)]
    # print greatest_val
    # print second_greatest
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0

    # print x




    #########################################
    return x
