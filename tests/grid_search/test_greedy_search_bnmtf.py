"""
Test the grid search for BNMTF, in grid_search_bnmtf.py
"""

from BNMTF.grid_search.greedy_search_bnmtf import GreedySearch
import numpy, pytest, random


def test_init():
    I,J = 10,9
    values_K = [1,2,4,5]
    values_L = [5,4,3]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaF':5, 'lambdaS':6, 'lambdaG':7 }
    initFG = 'exp'
    initS = 'random'
    iterations = 11
    
    greedysearch = GreedySearch(values_K,values_L,R,M,priors,initS,initFG,iterations)
    assert greedysearch.I == I
    assert greedysearch.J == J
    assert numpy.array_equal(greedysearch.values_K, values_K)
    assert numpy.array_equal(greedysearch.values_L, values_L)
    assert numpy.array_equal(greedysearch.R, R)
    assert numpy.array_equal(greedysearch.M, M)
    assert greedysearch.priors == priors
    assert greedysearch.iterations == iterations
    assert greedysearch.initS == initS
    assert greedysearch.initFG == initFG
    assert greedysearch.all_performances['BIC'] == []
    assert greedysearch.all_performances['AIC'] == []
    assert greedysearch.all_performances['loglikelihood'] == []
    
    
def test_search():
    # Check whether we get no exceptions...
    I,J = 10,9
    values_K = [1,2,4,5]
    values_L = [5,4,3]
    R = 2*numpy.ones((I,J))
    R[0,0] = 1
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaF':5, 'lambdaS':6, 'lambdaG':7 }
    initFG = 'exp'
    initS = 'exp'
    iterations = 1
    search_metric = 'BIC'
    
    numpy.random.seed(0)
    random.seed(0)
    greedysearch = GreedySearch(values_K,values_L,R,M,priors,initS,initFG,iterations)
    greedysearch.search(search_metric)
    
    with pytest.raises(AssertionError) as error:
        greedysearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."
    
    # We go from: (1,5) -> (1,4) -> (1,3), and try 6 locations
    assert len(greedysearch.all_values('BIC')) == 6
    
    
def test_all_values():
    I,J = 10,9
    values_K = [1,2,4,5]
    values_L = [5,4,3]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaF':5, 'lambdaS':6, 'lambdaG':7 }
    initFG = 'exp'
    initS = 'random'
    iterations = 11
    
    greedysearch = GreedySearch(values_K,values_L,R,M,priors,initS,initFG,iterations)
    greedysearch.all_performances = {
        'BIC' : [(1,2,10.),(2,2,20.),(2,3,30.),(2,4,40.),(5,3,20.)],
        'AIC' : [(1,2,10.),(2,2,20.),(2,3,30.),(2,4,25.),(5,3,20.)],
        'loglikelihood' : [(1,2,10.),(2,2,50.),(2,3,30.),(2,4,40.),(5,3,20.)]
    }
    assert numpy.array_equal(greedysearch.all_values('BIC'), [(1,2,10.),(2,2,20.),(2,3,30.),(2,4,40.),(5,3,20.)])
    assert numpy.array_equal(greedysearch.all_values('AIC'), [(1,2,10.),(2,2,20.),(2,3,30.),(2,4,25.),(5,3,20.)])
    assert numpy.array_equal(greedysearch.all_values('loglikelihood'), [(1,2,10.),(2,2,50.),(2,3,30.),(2,4,40.),(5,3,20.)])
    with pytest.raises(AssertionError) as error:
        greedysearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."
    
    
def test_best_value():
    I,J = 10,9
    values_K = [1,2,4,5]
    values_L = [5,4,3]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaF':5, 'lambdaS':6, 'lambdaG':7 }
    initFG = 'exp'
    initS = 'random'
    iterations = 11
    
    greedysearch = GreedySearch(values_K,values_L,R,M,priors,initS,initFG,iterations)
    greedysearch.all_performances = {
        'BIC' : [(1,2,10.),(2,2,20.),(2,3,30.),(2,4,40.),(5,3,20.)],
        'AIC' : [(1,2,10.),(2,2,20.),(2,3,30.),(2,4,25.),(5,3,20.)],
        'loglikelihood' : [(1,2,10.),(2,2,50.),(2,3,30.),(2,4,40.),(5,3,20.)]
    }
    assert greedysearch.best_value('BIC') == (2,4)
    assert greedysearch.best_value('AIC') == (2,3)
    assert greedysearch.best_value('loglikelihood') == (2,2)
    with pytest.raises(AssertionError) as error:
        greedysearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."