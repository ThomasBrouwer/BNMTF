"""
Test the class for Truncated Normal draws and expectations in truncated_normal.py.
"""
from BNMTF.code.distributions.truncated_normal import TruncatedNormal
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

    
# Test a draw - simply verify it is > 0.
# Also test whether we get inf for a very negative mean and high variance
def test_draw():
    mu = 1.0
    tau = 3.0
    tndist = TruncatedNormal(mu,tau)
    for i in range(0,100):
        assert tndist.draw() >= 0.0
        
    mu = -9999
    tau = 0.0001
    tndist = TruncatedNormal(mu,tau)
    for i in range(0,100):
        assert tndist.draw() == 0.0 # inf, so set to 0
    
    # Test everything is handled when tau = 0 - then draws should be inf, and hence return 0.0  
    mu = 0.32
    tau = 0.0
    tndist = TruncatedNormal(mu,tau)
    for i in range(0,100):
        assert tndist.draw() == 0.0