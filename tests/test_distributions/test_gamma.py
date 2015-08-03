"""
Test the class for Gamma draws and expectations in gamma.py.
"""
from BNMTF.code.distributions.gamma import Gamma

def test_class():
    alpha = 2.0
    beta = 3.0
    
    gammadist = Gamma(alpha,beta)
    assert gammadist.alpha == alpha
    assert gammadist.beta == beta
    
    expectation = 2.0 / 3.0
    assert gammadist.expectation() == expectation
    
    expectation_log = -0.67582795356964265 # digamma(2) - log_e (3) in Wolfram Alpha
    assert gammadist.expectation_log() == expectation_log