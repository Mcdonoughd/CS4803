import numpy as np
from problem3 import random_walk
# -------------------------------------------------------------------------
'''
    Problem 4: Solving sink-node problme in PageRank
    In this problem, we implement the pagerank algorithm which can solve the sink node problem.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''


# --------------------------
def compute_S(A):
    '''
        compute the transition matrix S from addjacency matrix A, which solves sink node problem by filling the all-zero columns in A.
        S[j][i] represents the probability of moving from node i to node j.
        If node i is a sink node, S[j][i] = 1/n.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                S: transition matrix, a (n by n) numpy matrix of float values.  S[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix S should sum to 1.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    S = np.zeros(shape=(len(A), len(A[0])))
    # for all columns in A
    for j in range(len(A[0])):
        # count the total connection
        count = 0.0
        for i in range(len(A)):
            count += A[i][j]
        # find probability of one connection
        #print count
        if count != 0:
            prob = 1 / count
            # populate P with probabilities
            for g in range(len(A)):
                if A[g][j] != 0:
                    S[g][j] = prob
        #print count == 0.0
        if count == 0:
            prob = 1/float(len(S))
            #print prob
            for k in range(len(S)):
                S[k][j] = prob
                #print S

    #########################################
    return S


# --------------------------
def pagerank_v2(A):
    ''' 
        A simplified version of PageRank algorithm, which solves the sink node problem.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the ranking scores, a numpy vector of float values, such as np.array([[.3], [.5], [.7]])
    '''

    # Initialize the score vector with all one values
    num_nodes, _ = A.shape  # get the number of nodes (n)
    x_0 = np.ones((num_nodes, 1))  # create an all-one vector of shape (n by 1)

    # compute the transition matrix from adjacency matrix
    S = compute_S(A)

    # random walk
    x, n_steps = random_walk(S, x_0)

    return x
