"""
Test the class for Truncated Normal draws and expectations in truncated_normal_vector.py.
"""
from BNMTF.code.distributions.truncated_normal_vector import TN_vector_draw, TN_vector_expectation, TN_vector_variance, TN_vector_mode
from scipy.stats import norm
import numpy

def test_expectation():
    # One normal case, one exponential approximation
    mu = [1.0, -1]
    tau = [3.0, 2000]
    sigma = [0.5773502691896258,0.022360679774997897]
    
    lambdav = ( norm.pdf( - mu[0] / sigma[0] ) ) / ( 1 - norm.cdf( - mu[0] / sigma[0] ) )
    expectation = mu[0] + sigma[0] * lambdav
    assert numpy.array_equal(TN_vector_expectation(mu,tau), [expectation, 1./2000.])
    
def test_variance():
    # One normal case, one exponential approximation
    mu = [1.0, -1]
    tau = [3.0, 2000]
    sigma = [0.5773502691896258,0.022360679774997897]
    
    lambdav = ( norm.pdf( - mu[0] / sigma[0] ) ) / ( 1 - norm.cdf( - mu[0] / sigma[0] ) )
    variance = sigma[0]**2 * ( 1 - ( lambdav * ( lambdav + mu[0] / sigma[0] ) ) )
    assert numpy.array_equal(TN_vector_variance(mu,tau), [variance, (1./2000.)**2])

# Test a draw - simply verify it is > 0.
# Also test whether we get inf for a very negative mean and high variance
def test_draw():
    # One normal case, and one when tau=0 - then draws should be inf, and hence return 0.0  
    mu = [1.0, 0.32]
    tau = [3.0, 0.0]
    for i in range(0,100):
        v1,v2 = TN_vector_draw(mu,tau)
        assert v1 >= 0.0 and v2 == 0.0
        
# Test the mode
def test_mode():
    # Positive mean
    mus = [1.0, -2.0]
    assert numpy.array_equal(TN_vector_mode(mus), [1.0, 0.0])