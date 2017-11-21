import numpy as np

# -------------------------------------------------------------------------
'''
    Problem 3: PageRank algorithm (version 1) 
    In this problem, we implement a simplified version of the pagerank algorithm, which doesn't consider about sink node problem or sink region problem.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''


# --------------------------
def compute_P(A):
    '''
        compute the transition matrix P from addjacency matrix A. P[j][i] represents the probability of moving from node i to node j.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
    The values in each column of matrix P should sum to 1.
    '''
    #########################################
    ## INSER YOUR CODE HERE
    p = np.zeros(shape=(len(A), len(A[0])))
    # for all columns in A
    for j in range(len(A[0])):
        # count the total connection
        count = 0
        for i in range(len(A)):
            count += A[i][j]
        # find probability of one connection
        prob = 1 / count
        # populate P with probabilities
        for g in range(len(A)):

            if A[g][j] != 0:
                p[g][j] = prob

    #########################################
    return p


# --------------------------
def random_walk_one_step(P, x_i):
    '''
        compute the result of one step random walk.
        Input: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                x_i: pagerank scores before the i-th step of random walk. a numpy vector of shape (n by 1).
        Output: 
                x_i_plus_1: pagerank scores after the i-th step of random walk. a numpy vector of shape (n by 1).
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    x_i_plus_1 = np.matmul(P, x_i)
    #########################################
    return x_i_plus_1


# --------------------------
def random_walk(P, x_0, max_steps=10000):
    '''
        compute the result of multiple-step random walk. The random walk should stop if the score vector x no longer change (converge) after one step of random walk, or the number of iteration reached max_steps.
        Input: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                x_0: the initial pagerank scores. a numpy vector of shape (n by 1).
                max_steps: the maximium number of random walk steps. an integer value.  
        Output: 
                x: the final pagerank scores after multiple steps of random walk. a numpy vector of shape (n by 1).
                n_steps: the number of steps actually used (for example, if the vector x no longer changes after 3 steps of random walk, return the value 3. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    n_steps = 0
    for i in range(max_steps):

        # print prev_step
        if n_steps == 0:
            x = random_walk_one_step(P, x_0)
            prev_step = x_0
            if np.allclose(prev_step, x):
                # print n_steps
                n_steps += 1
                return x, n_steps
        else:
            prev_step = x
            x = random_walk_one_step(P, x)
        # print x
        n_steps += 1
        if np.allclose(prev_step, x):
            # print n_steps
            print n_steps
            return x, n_steps

    #########################################
    return x, n_steps


# --------------------------
def pagerank_v1(A):
    ''' 
        A simplified version of PageRank algorithm.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Here we ignore the issues of sink nodes and sink regions in the network.
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the ranking scores, a numpy vector of float values, such as np.array([[.3], [.5], [.7]])
    '''

    # compute the transition matrix from adjacency matrix
    P = compute_P(A)

    # initialize the score vector with all one values
    num_nodes, _ = A.shape  # get the number of nodes (n)
    x_0 = np.ones((num_nodes, 1))  # create an all-one vector of shape (n by 1)

    # random walk
    x, n_steps = random_walk(P, x_0)
    return x
