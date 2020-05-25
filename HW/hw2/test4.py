from problem4 import *
from io import BytesIO

'''
    Unit test 4:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests test4.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_parse_line():
    ''' test the correctness of parse_line() function'''
    
    # an example line
    line = 'A,1,2,1.5,3,4'
    
    # call the function
    matrix_name, i,j,v, nr, nc = MatMul.parse_line(line)
    
    # test the correctness of the result
    assert matrix_name =='A'
    assert type(i) == int
    assert type(j) == int
    assert type(v) == float 
    assert type(nr) == int
    assert type(nc) == int
    assert i == 1
    assert j == 2
    assert v == 1.5 
    assert nr == 3
    assert nc == 4

#-------------------------------------------------------------------------
def test_mapper():
    ''' test the correctness of mapper() function'''

    # create an object
    m = MatMul()

    # call the function
    outs = m.mapper(None,'A,1,2,1.5,1,2') 

    # read one key-value pair from the output generator 
    key, value = outs.next()

    # test whether or not the returned key is a tuple 
    assert type(key) == tuple 
    assert len(key) ==3
    assert key == ('C',1,1)

    # test whether or not the returned value is a tuple 
    assert type(value) == tuple 
    assert len(value) == 4
    assert value == ('A',1,2,1.5)
    
    # read one key-value pair from the output generator 
    key, value = outs.next()
    assert key == ('C',1,2)
    assert value == ('A',1,2,1.5)

    #-------------------------
    # test another example
    
    # call the function
    outs = m.mapper(None,'B,2,1,1.7,2,1') 

    # read one key-value pair from the output generator 
    key, value = outs.next()
    assert key == ('C',1,1)
    assert value == ('B',2,1,1.7)
    
    # read one key-value pair from the output generator 
    key, value = outs.next()
    assert key == ('C',2,1)
    assert value == ('B',2,1,1.7)


#-------------------------------------------------------------------------
def test_reducer():
    ''' test the correctness of reducer() function'''
   
    # create an object
    m = MatMul()
    in_key = ('C',1,1)
    values = [  ('A',1,1,1.0),
                ('A',1,2,2.0),
                ('B',1,1,3.0),
                ('B',2,1,4.0) ] 

    # convert the value list into a generator
    values = (i for i in values) 

    # call the function
    out_key, value = m.reducer(in_key, values).next()

    assert out_key == ('C', 1, 1)
    assert type(value) == float 
    assert value == 11.0

    
#-------------------------------------------------------------------------
def test_MatMul():
    # cast lines of text into standard Input
    data = open('matrix.csv','rb')
    stdin = BytesIO(data.read())

    # create a map reduce job (in a sandbox)
    job = MatMul()
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
        results.append([key,value])

    # test the correctness of the result
    assert len(results) == 9 # C is a 3 by 3 matrix
    test = False
    for r in results:
        if r[0] == ['C',1,1]:
            test = True
            assert type(r[1]) == float
            assert r[1] == 61. 

    assert test


