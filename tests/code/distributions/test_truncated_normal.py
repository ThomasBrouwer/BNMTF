"""
Test the class for Truncated Normal draws and expectations in truncated_normal.py.
"""
from BNMTF.code.distributions.truncated_normal import TN_draw, TN_expectation, TN_variance, TN_mode
from scipy.stats import norm
import numpy

def test_expectation():
    mu = 1.0
    tau = 3.0
    sigma = 0.5773502691896258
    
    lambdav = ( norm.pdf( - mu / sigma ) ) / ( 1 - norm.cdf( - mu / sigma ) )
    expectation = mu + sigma * lambdav
    assert TN_expectation(mu,tau) == expectation
    
    # Also test that we get variance and exp of an Exp if mu is less than -30*sigma
    mu = -1.
    tau = 2000.
    assert TN_expectation(mu,tau) == 1./2000.
    
    
def test_variance():
    mu = 1.0
    tau = 3.0
    sigma = 0.5773502691896258
    
    lambdav = ( norm.pdf( - mu / sigma ) ) / ( 1 - norm.cdf( - mu / sigma ) )
    variance = sigma**2 * ( 1 - ( lambdav * ( lambdav + mu / sigma ) ) )
    assert TN_variance(mu,tau) == variance

    # Also test that we get variance and exp of an Exp if mu is less than -30*sigma
    mu = -1.
    tau = 2000.
    assert TN_variance(mu,tau) == (1./2000.)**2
    
# Test a draw - simply verify it is > 0.
# Also test whether we get inf for a very negative mean and high variance
def test_draw():
    mu = 1.0
    tau = 3.0
    for i in range(0,100):
        assert TN_draw(mu,tau) >= 0.0
    
    # Test everything is handled when tau = 0 - then draws should be inf, and hence return 0.0  
    mu = 0.32
    tau = 0.0
    for i in range(0,100):
        assert TN_draw(mu,tau) == 0.0
        
# Test the mode
def test_mode():
    # Positive mean
    mu = 1.0
    assert TN_mode(mu) == mu
    
    # Negative mean
    mu = -2.0
    assert TN_mode(mu) == 0.