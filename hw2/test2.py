from problem2 import *

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_import_W():
    ''' test the correctness of import_W() function'''

    # call the function
    W = import_W() 

    # test whether or not W is a numpy array
    assert type(W) == np.ndarray 

    # test the shape of the matrix 
    assert W.shape == (11621, 2)

    # test the dtype of the matrix
    assert W.dtype == int    

    # test the correctness of the result
    assert W[0][0] == 0 
    assert W[0][1] == 1 
    assert W[-1][0] == 667
    assert W[-1][1] == 286

#-------------------------------------------------------------------------
def test_import_team_names():
    ''' test the correctness of import_team_names() function'''

    # call the function
    team_names = import_team_names() 

    # test whether or not team_names is a python list
    assert type(team_names) == list

    # test the correctness of the result
    assert team_names[0] == 'Liberty'
    assert team_names[1] == 'Randolph Col'
    assert team_names[-1] == 'York (NE)'



#-------------------------------------------------------------------------
def test_team_rating():
    ''' test the correctness of team_rating() function'''
    # call the function
    top_teams,top_ratings = team_rating() 

    # test whether or not top_teams is a python list
    assert type(top_teams) == list
    assert type(top_ratings) == list
    assert len(top_teams) == 1031
    assert len(top_ratings) == 1031


    # test the correctness of the result
    assert top_teams[:3] ==['Villanova', 'Kansas', 'Kentucky']
    assert np.allclose(top_ratings[:3], [657.5686931994142, 650.2020558804585, 636.3019699432945])

