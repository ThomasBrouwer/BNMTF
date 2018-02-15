"""
Test the class for Gamma draws and expectations in gamma.py.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

from BNMTF.code.models.distributions.gamma import gamma_draw, gamma_expectation, gamma_expectation_log, gamma_mode

def test_expectation():
    alpha = 2.0
    beta = 3.0
    
    expectation = 2.0 / 3.0
    assert gamma_expectation(alpha,beta) == expectation
    
def test_expectation_log():
    alpha = 2.0
    beta = 3.0
    
    expectation_log = -0.67582795356964265 # digamma(2) - log_e (3) in Wolfram Alpha
    assert gamma_expectation_log(alpha,beta) == expectation_log
    
# Test a draw - simply verify it is > 0.
def test_draw():
    alpha = 2.0
    beta = 3.0
    for i in range(0,100):
        assert gamma_draw(alpha,beta) >= 0.0
        
# Test median
def test_median():
    alpha = 2.0
    beta = 3.0
    median = 1./3.
    assert gamma_mode(alpha,beta) == median