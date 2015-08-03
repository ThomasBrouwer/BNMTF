"""
Test the class for Truncated Normal draws and expectations in truncated_normal.py.
"""
from BNMTF.code.truncated_normal import TruncatedNormal
from scipy.stats import norm

def test_class():
    mu = 1.0
    tau = 3.0
    sigma = 0.5773502691896258
    
    tndist = TruncatedNormal(mu,tau)
    assert tndist.mu == mu
    assert tndist.tau == tau
    assert tndist.sigma == sigma
    assert tndist.a == - mu / sigma
    assert tndist.b == float("inf")
    
    lambdav = ( norm.pdf( - mu / sigma ) ) / ( 1 - norm.cdf( - mu / sigma ) )
    expectation = mu + sigma * lambdav
    assert tndist.expectation() == expectation
    
    variance = sigma**2 * ( 1 - ( lambdav * ( lambdav + mu / sigma ) ) )
    assert tndist.variance() == variance