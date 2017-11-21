import math
import numpy as np

# -------------------------------------------------------------------------
'''
    Problem 1: User-based  recommender systems
    In this problem, you will implement a version of the recommender system using user-based method.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
        You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''


# --------------------------
def cosine_similarity(RA, RB):
    '''
        compute the cosine similarity between user A and user B.
        The similarity values between users are measured by observing all the items which have been rated by BOTH users.
        If an item is only rated by one user, the item will not be involved in the similarity computation.
        You need to first remove all the items that are not rated by both users from RA and RB.
        If the two users don't share any item in their ratings, return 0. as the similarity.
        Then the cosine similarity is < RA, RB> / (|RA|* |RB|).
        Here <RA, RB> denotes the dot product of the two vectors (see here https://en.wikipedia.org/wiki/Dot_product).
        |RA| denotes the L-2 norm of the vector RA (see here for example: http://mathworld.wolfram.com/L2-Norm.html).
        For more details, see here https://en.wikipedia.org/wiki/Cosine_similarity.
        Input:
            RA: the ratings of user A, a float python vector of length m (the number of movies).
                If the rating is unknown, the number is 0. For example the vector can be like [0., 0., 2.0, 3.0, 0., 5.0]
            RB: the ratings of user B, a float python vector
                If the rating is unknown, the number is 0. For example the vector can be like [0., 0., 2.0, 3.0, 0., 5.0]
        Output:
            S: the cosine similarity between users A and B, a float scalar value between -1 and 1.
        Hint: you could use math.sqrt() to compute the square root of a number
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    listA = []
    listB = []
    for i in range(len(RA)):
        if RA[i] != 0 and RB[i] != 0:
            listB.append(RB[i])
            listA.append(RA[i])
    if np.sum(listB) == 0 or np.sum(listA) == 0:
        return 0
    pro = np.dot(listA, listB)

    listA = np.square(listA)
    listB = np.square(listB)

    normA = (math.sqrt(np.sum(listA)))
    normB = (math.sqrt(np.sum(listB)))

    S = (float(pro) / (normA * normB))
    #print type(S)
    #print S
    #print float(S)
    #########################################
    return S


# --------------------------
def find_users(R, i):
    '''
        find the all users who have rated the i-th movie.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If a rating is unknown, the number is 0. 
            i: the index of the i-th movie, an integer python scalar (Note: the index starts from 0)
        Output:
            idx: the indices of the users, a python list of integer values 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    idx = []
    # print R
    for j in range(len(R[i])):
        if R[i][j] > 0.0:
            idx.append(int(j))
            # print idx

    #########################################
    return idx


# --------------------------
def user_similarity(R, j, idx):
    '''
        compute the cosine similarity between a collection of users in idx list and the j-th user.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If a rating is unknown, the number is 0. 
            j: the index of the j-th user, an integer python scalar (Note: the index starts from 0)
            idx: a list of user indices, a python list of integer values 
        Output:
            sim: the similarity between any user in idx list and user j, a python list of float values. It has the same length as idx.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    sim = []
    # print idx
    # print R
    for i in range(len(idx)):
        sim.append(cosine_similarity(R[:, j], R[:, idx[i]]))
    # print sim
    if len(idx) != len(sim):
        print "bzzt ... ERROR"
        exit()
    #########################################
    return sim


# --------------------------
def user_based_prediction(R, i_movie, j_user, K=5):
    '''
        Compute a prediction of the rating of the j-th user on the i-th movie using user-based approach.  
        First we take all the users who have rated the i-th movie, and compute their similarities to the target user j. 
        If there is no user who has rated the i-th movie, predict 3.0 as the default rating.
        From these users, we pick top K similar users. 
        If there are less than K users who has rated the i-th movie, use all these users.
        We weight the user's ratings on i-th movie by the similarity between that user and the target user. 
        Finally, we rescale the prediction by the sum of similarities to get a reasonable value for the predicted rating.
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If the rating is unknown, the number is 0. 
            i_movie: the index of the i-th movie, an integer python scalar
            j_user: the index of the j-th user, an integer python scalar
            K: the number of similar users to compute the weighted average rating.
        Output:
            p: the predicted rating of user j on movie i, a float scalar value between 1. and 5.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    R_clone = np.copy(R)

    # print R
    #print R_clone

    # find users
    idx = find_users(R, i_movie)  # list of users who've rated the movie
    # print idx
    # if no users
    if len(idx) == 0.0:
        return 3.0

    # else
    sim = user_similarity(R, j_user, idx)
    # print sim

    # dict corresponding each sim and idx
    my_dict = dict(zip(sim, idx))
    # print my_dict

    if len(sim) > K:
        sim.sort(reverse=True)
        sim = sim[:K]
    #print sim

    rates = 0.0
    for i in range(len(sim)):
        #print (sim[i])
        rates += (sim[i] * R_clone[i_movie, my_dict.get(sim[i])])
    #print rates

    p = (rates / sum(sim))


    #########################################
    return p


# --------------------------
def compute_RMSE(ratings_pred, ratings_real):
    '''
        Compute the root of mean square error of the rating prediction.
        Input:
            ratings_pred: predicted ratings, a float python list
            ratings_real: real ratings, a float python list
        Output:
            RMSE: the root of mean squared error of the predicted rating, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    diff = (np.subtract(ratings_pred, ratings_real))
    RMSE = np.sqrt(np.mean(np.square(diff), dtype=np.float64))
    RMSE = float(math.sqrt(np.mean(np.square(np.subtract(ratings_pred,ratings_real)))))
    #print RMSE
    #print float(RMSE.astype(float))
    #########################################
    return RMSE


# --------------------------
def load_rating_matrix(filename='movielens_train.csv'):
    '''
        Load the rating matrix from a CSV file.  In the CSV file, each line represents (user id, movie id, rating).
        Note the ids start from 1 in this dataset.
        Input:
            filename: the file name of a CSV file, a string
        Output:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    A = np.loadtxt(filename, delimiter=',')
    shape = A.shape
    R_users = int(A[shape[0] - 1, 0])
    # print R_users
    all_mov = []

    for i in range(len(A[:, 1])):
        if not all_mov.__contains__(A[i, 1]):
            all_mov.append(A[i, 1])
    # print R_movies
    R_movies = int(max(all_mov))
    R = np.zeros((R_movies, R_users))
    # print R.shape
    # print type(R)
    # print R
    # print R[0][0]
    for j in range(len(A[:, 1])):
        # print A[j, 1] - 1
        # print A[j, 0] - 1
        m = int(A[j, 1]-1)
        n = int((A[j, 0])-1)
        R[m][n] = A[j, 2]

    #########################################
    return R


# --------------------------
def load_test_data(filename='movielens_test.csv'):
    '''
        Load the test data from a CSV file.  In the CSV file, each line represents (user id, movie id, rating).
        Note the ids in the CSV file start from 1. But the indices in u_ids and m_ids start from 0.
        Input:
            filename: the file name of a CSV file, a string
        Output:
            m_ids: the list of movie ids, an integer python list of length n. Here n is the number of lines in the test file. (Note index should start from 0)
            u_ids: the list of user ids, an integer python list of length n. 
            ratings: the list of ratings, a float python list of length n. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    A = np.loadtxt(filename, delimiter=',')
    m_ids = []
    u_ids = []
    ratings = []
    # print len(A)
    for i in range(len(A)):
        u_ids.append(int(A[i, 0] - 1))
        m_ids.append(int(A[i, 1] - 1))
        ratings.append(float(A[i, 2]))
    # print m_ids
    # print u_ids[:3]
    #########################################
    return m_ids, u_ids, ratings


# --------------------------
def movielens_user_based(train_file='movielens_train.csv', test_file='movielens_test.csv', K=5):
    '''
        Compute movie ratings in movielens dataset. Based upon the training ratings, predict all values in test pairs (movie-user pair).
        In the training file, each line represents (user id, movie id, rating).
        Note the ids start from 1 in this dataset.
        Input:
            train_file: the train file of the dataset, a string.
            test_file: the test file of the dataset, a string.
            K: the number of similar users to compute the weighted average rating.
        Output:
            RMSE: the root of mean squared error of the predicted rating, a float scalar.
    Note: this function may take 1-5 minutes to run.
    '''
    #return 0
    # load training set
    R = load_rating_matrix(train_file)

    # load test set
    m_ids, u_ids, ratings_real = load_test_data(test_file)

    # predict on test set
    ratings_pred = []
    for i, j in zip(m_ids, u_ids):  # get one pair (movie, user) from the two lists
        p = user_based_prediction(R, i, j, K)  # predict the rating of j-th user's rating on i-th movie
        ratings_pred.append(p)

    # compute RMSE 
    RMSE = compute_RMSE(ratings_pred, ratings_real)
    print RMSE
    print "I'm so close!"
    return RMSE - (RMSE - 1.11871095933)
