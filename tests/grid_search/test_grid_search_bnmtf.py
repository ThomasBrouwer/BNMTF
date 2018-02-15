"""
Test the grid search for BNMTF, in grid_search_bnmtf.py
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from BNMTF.code.cross_validation.grid_search_bnmtf import GridSearch
from BNMTF.code.models.bnmtf_vb_optimised import bnmtf_vb_optimised
import numpy, pytest

classifier = bnmtf_vb_optimised

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
    
    gridsearch = GridSearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations)
    assert gridsearch.I == I
    assert gridsearch.J == J
    assert numpy.array_equal(gridsearch.values_K, values_K)
    assert numpy.array_equal(gridsearch.values_L, values_L)
    assert numpy.array_equal(gridsearch.R, R)
    assert numpy.array_equal(gridsearch.M, M)
    assert gridsearch.priors == priors
    assert gridsearch.iterations == iterations
    assert gridsearch.initS == initS
    assert gridsearch.initFG == initFG
    assert gridsearch.all_performances['BIC'].shape == (4,3)
    assert gridsearch.all_performances['AIC'].shape == (4,3)
    assert gridsearch.all_performances['loglikelihood'].shape == (4,3)
    
    
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
    initS = 'random'
    iterations = 1
    
    gridsearch = GridSearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations)
    gridsearch.search()
    
    
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
    
    gridsearch = GridSearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations)
    gridsearch.all_performances = {
        'BIC' : [[10,9,8],[11,12,13],[17,16,15],[13,13,13]],
        'AIC' : [[8,8,8],[7,7,7],[10,11,15],[6,6,6]],
        'loglikelihood' : [[10,12,13],[17,18,29],[5,4,3],[3,2,1]]
    }
    assert numpy.array_equal(gridsearch.all_values('BIC'), [[10,9,8],[11,12,13],[17,16,15],[13,13,13]])
    assert numpy.array_equal(gridsearch.all_values('AIC'), [[8,8,8],[7,7,7],[10,11,15],[6,6,6]])
    assert numpy.array_equal(gridsearch.all_values('loglikelihood'), [[10,12,13],[17,18,29],[5,4,3],[3,2,1]])
    with pytest.raises(AssertionError) as error:
        gridsearch.all_values('FAIL')
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
    
    gridsearch = GridSearch(classifier,values_K,values_L,R,M,priors,initS,initFG,iterations)
    gridsearch.all_performances = {
        'BIC' : [[10,9,8],[11,12,13],[17,16,15],[13,13,13]],
        'AIC' : [[8,8,8],[7,7,7],[10,11,15],[6,5,6]],
        'loglikelihood' : [[10,12,13],[17,18,29],[5,4,3],[3,2,1]]
    }
    assert gridsearch.best_value('BIC') == (1,3)
    assert gridsearch.best_value('AIC') == (5,4)
    assert gridsearch.best_value('loglikelihood') == (5,3)
    with pytest.raises(AssertionError) as error:
        gridsearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."