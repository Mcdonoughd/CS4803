from problem3 import *
from io import BytesIO

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_mapper():
    ''' test the correctness of mapper() function'''

    # create an object
    w = WordCount()

    # call the function
    outs = w.mapper(None,'hello world') 

    # read one key-value pair from the output generator 
    out_pair = outs.next()

    # test whether or not the returned count is an integer
    assert type(out_pair[1]) == int

    # test the correctness of the result
    assert out_pair[1] == 11

    #-------------------------
    # test another example

    # call the function
    outs = w.mapper(None,'hello') 

    # test the correctness of the result
    out_pair = outs.next()
    assert out_pair[1] == 5 

#-------------------------------------------------------------------------
def test_reducer():
    ''' test the correctness of reducer() function'''

    # create an object
    w = WordCount()

   # call the function
    outs = w.reducer(None,[1,2,3]) 

    # read one key-value pair from the output generator 
    out_pair = outs.next()

    # test whether or not the returned count is an integer
    assert type(out_pair[1]) == int

    # test the correctness of the result
    assert out_pair[1] == 6

    #-------------------------
    # test another example

    # call the function
    outs = w.reducer(None,[1,2,3,4]) 

    # test the correctness of the result
    out_pair = outs.next()
    assert out_pair[1] == 10 



#-------------------------------------------------------------------------
def test_WordCount():
    # cast lines of text into standard Input
    num_lines = 5
    stdin = BytesIO(b'hello\n' * num_lines)

    # create a map reduce job (in a sandbox)
    job = WordCount()
    job.sandbox(stdin=stdin)

    # create a job runner
    runner = job.make_runner()

    # run the mapreduce job 
    runner.run()

    # parse the outputs
    results = []
    for line in runner.stream_output():
        # Use the job's specified protocol to read an output key-value pair
        key, value = job.parse_output_line(line)
        # append the value into a list 
        results.append(value)

    # test the correctness of the result
    assert results ==[25]



