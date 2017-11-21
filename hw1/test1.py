from problem1 import *

'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''


#-------------------------------------------------------------------------
def test_swap():
    ''' test the correctness of swap() function in problem1.py'''
    A = [1,2,3] # create a list example
    swap(A,0,2) # call the swap function

    # test whether or not A is a python array
    assert type(A) == list

    # check whether or not the two elements in the list are switched
    assert A[0]== 3 
    assert A[2]== 1


def test_bubblesort():
    ''' test the correctness of bubblesort() function in problem1.py'''
    A = [8,5,3,1,9,6,0,7,4,2,5] # create a list example
    bubblesort(A) # call the function

    # test whether or not A is a python array
    assert type(A) == list

    # check whether or not the list is sorted 
    assert A == [0,1,2,3,4,5,5,6,7,8,9]
    
