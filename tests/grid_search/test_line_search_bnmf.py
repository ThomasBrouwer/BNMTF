"""
Test the line search for BNMF, in line_search_bnmf.py
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from BNMTF.code.cross_validation.line_search_bnmf import LineSearch
from BNMTF.code.models.bnmf_vb_optimised import bnmf_vb_optimised
import numpy, pytest

classifier = bnmf_vb_optimised

def test_init():
    I,J = 10,9
    values_K = [1,2,4,5]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaU':5, 'lambdaV':6 }
    initUV = 'exp'
    iterations = 11
    
    linesearch = LineSearch(classifier,values_K,R,M,priors,initUV,iterations)
    assert linesearch.I == I
    assert linesearch.J == J
    assert numpy.array_equal(linesearch.values_K, values_K)
    assert numpy.array_equal(linesearch.R, R)
    assert numpy.array_equal(linesearch.M, M)
    assert linesearch.priors == priors
    assert linesearch.iterations == iterations
    assert linesearch.initUV == initUV
    assert linesearch.all_performances == { 'BIC' : [], 'AIC' : [], 'loglikelihood' : [], 'MSE' : [], 'ELBO' : [] }
    
    
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
    
    linesearch = LineSearch(classifier,values_K,R,M,priors,initUV,iterations)
    linesearch.search()
    
    
def test_all_values():
    I,J = 10,9
    values_K = [1,2,4,5]
    R = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    priors = { 'alpha':3, 'beta':4, 'lambdaU':5, 'lambdaV':6 }
    initUV = 'exp'
    iterations = 11
    
    linesearch = LineSearch(classifier,values_K,R,M,priors,initUV,iterations)
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
    
    linesearch = LineSearch(classifier,values_K,R,M,priors,initUV,iterations)
    linesearch.all_performances = {
        'BIC' : [10,9,8,7],
        'AIC' : [11,13,12,14],
        'loglikelihood' : [16,15,18,17],
        'MSE' : [16,20,18,17]
    }
    assert linesearch.best_value('BIC') == 5
    assert linesearch.best_value('AIC') == 1
    assert linesearch.best_value('loglikelihood') == 2
    assert linesearch.best_value('MSE') == 1
    with pytest.raises(AssertionError) as error:
        linesearch.all_values('FAIL')
    assert str(error.value) == "Unrecognised metric name: FAIL."