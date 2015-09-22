"""
Test the line search for BNMF, in line_search_bnmf.py
"""

from BNMTF.grid_search.line_search_bnmf import LineSearch
import numpy, pytest


def test_init():
    I,J = 10,9
    values_K = [1,2,4,5]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaU':5, 'lambdaV':6 }
    initUV = 'exp'
    iterations = 11
    
    linesearch = LineSearch(values_K,R,M,priors,initUV,iterations)
    assert linesearch.I == I
    assert linesearch.J == J
    assert numpy.array_equal(linesearch.values_K, values_K)
    assert numpy.array_equal(linesearch.R, R)
    assert numpy.array_equal(linesearch.M, M)
    assert linesearch.priors == priors
    assert linesearch.iterations == iterations
    assert linesearch.initUV == initUV
    assert linesearch.all_performances == { 'BIC' : [], 'AIC' : [], 'loglikelihood' : [] }
    
    
def test_search():
    # Check whether we get no exceptions...
    I,J = 10,9
    values_K = [1,2,4,5]
    R = 2*numpy.ones((I,J))
    R[0,0] = 1
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaU':5, 'lambdaV':6 }
    initUV = 'exp'
    iterations = 1
    
    linesearch = LineSearch(values_K,R,M,priors,initUV,iterations)
    linesearch.search()
    
    
def test_all_values():
    I,J = 10,9
    values_K = [1,2,4,5]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaU':5, 'lambdaV':6 }
    initUV = 'exp'
    iterations = 11
    
    linesearch = LineSearch(values_K,R,M,priors,initUV,iterations)
    linesearch.all_performances = {
        'BIC' : [10,9,8,7],
        'AIC' : [11,13,12,14],
        'loglikelihood' : [16,15,18,17]
    }
    assert numpy.array_equal(linesearch.all_values('BIC'), [10,9,8,7])
    assert numpy.array_equal(linesearch.all_values('AIC'), [11,13,12,14])
    assert numpy.array_equal(linesearch.all_values('loglikelihood'), [16,15,18,17])
    with pytest.raises(AssertionError) as error:
        linesearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."
    
    
def test_best_value():
    I,J = 10,9
    values_K = [1,2,4,5]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaU':5, 'lambdaV':6 }
    initUV = 'exp'
    iterations = 11
    
    linesearch = LineSearch(values_K,R,M,priors,initUV,iterations)
    linesearch.all_performances = {
        'BIC' : [10,9,8,7],
        'AIC' : [11,13,12,14],
        'loglikelihood' : [16,15,18,17]
    }
    assert linesearch.best_value('BIC') == 1
    assert linesearch.best_value('AIC') == 5
    assert linesearch.best_value('loglikelihood') == 4
    with pytest.raises(AssertionError) as error:
        linesearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."