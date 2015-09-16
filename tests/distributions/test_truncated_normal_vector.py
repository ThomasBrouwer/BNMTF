"""
Test the class for Truncated Normal draws and expectations in truncated_normal_vector.py.
"""
from BNMTF.code.distributions.truncated_normal_vector import TruncatedNormalVector
from scipy.stats import norm
import numpy

def test_class():
    # One normal case, one Exponential approx case
    mu = [1.0, -1]
    tau = [3.0, 2000]
    sigma = [0.5773502691896258,0.022360679774997897]
    
    tndist = TruncatedNormalVector(mu,tau)
    assert numpy.array_equal(tndist.mu, mu)
    assert numpy.array_equal(tndist.tau, tau)
    assert numpy.array_equal(tndist.sigma, sigma)
    assert numpy.array_equal(tndist.a, - numpy.array(mu) / sigma)
    assert numpy.array_equal(tndist.b, [float("inf"),float("inf")])
    
    lambdav = ( norm.pdf( - mu[0] / sigma[0] ) ) / ( 1 - norm.cdf( - mu[0] / sigma[0] ) )
    expectation = mu[0] + sigma[0] * lambdav
    assert numpy.array_equal(tndist.expectation(), [expectation, 1./2000.])
    
    variance = sigma[0]**2 * ( 1 - ( lambdav * ( lambdav + mu[0] / sigma[0] ) ) )
    assert numpy.array_equal(tndist.variance(), [variance, (1./2000.)**2])

# Test a draw - simply verify it is > 0.
# Also test whether we get inf for a very negative mean and high variance
def test_draw():
    # One normal case, and one when tau=0 - then draws should be inf, and hence return 0.0  
    mu = [1.0, 0.32]
    tau = [3.0, 0.0]
    tndist = TruncatedNormalVector(mu,tau)
    for i in range(0,100):
        v1,v2 = tndist.draw()
        assert v1 >= 0.0 and v2 == 0.0