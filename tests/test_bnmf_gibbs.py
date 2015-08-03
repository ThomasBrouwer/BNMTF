"""
Tests for the BNMF Gibbs sampler.
"""

import numpy, math, pytest, itertools, random
from BNMTF.code.bnmf_gibbs import bnmf_gibbs


""" Test constructor """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    K = 0
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R1,M,K)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R2,M,K)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R3,M,K)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Test getting an exception if a row or column is entirely unknown
    R = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R,M1,K)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R,M2,K)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    
""" Test initialing parameters """
def test_initialise():
    I,J,K = 5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    
    lambda_U = numpy.ones((I,K))
    lambda_V = numpy.ones((J,K))
    alpha, beta = 3, 1
    
    BNMF = bnmf_gibbs(R,M,K)
    BNMF.initialise(lambda_U,lambda_V,alpha,beta)
    
    # Only assert all values are non-negative - not much else we can do
    assert BNMF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMF.U[i,k] >= 0.0
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert BNMF.V[j,k] >= 0.0