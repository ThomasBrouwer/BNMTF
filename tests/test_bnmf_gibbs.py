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
    I,J,K = 5,3,0
    lambdaU = numpy.ones((I,K))
    lambdaV = numpy.ones((J,K))
    alpha, beta = 3, 1    
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R1,M,K,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R2,M,K,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R3,M,K,priors)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Test getting an exception if a row or column is entirely unknown
    R4 = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R4,M1,K,priors)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmf_gibbs(R4,M2,K,priors)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    M = numpy.ones((I,J))
    BNMF = bnmf_gibbs(R5,M,K,priors)
    
    assert numpy.array_equal(BNMF.R,R5)
    assert numpy.array_equal(BNMF.M,M)
    assert BNMF.I == I
    assert BNMF.J == J
    assert BNMF.K == K
    assert BNMF.size_Omega == I*J
    assert BNMF.alpha == alpha
    assert BNMF.beta == beta
    assert numpy.array_equal(BNMF.lambdaU,lambdaU)
    assert numpy.array_equal(BNMF.lambdaV,lambdaV)
    
    
""" Test initialing parameters """
def test_initialise():
    I,J,K = 5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    # First do a random initialisation - we can then only check whether values are correctly initialised
    init = 'random'
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    
    assert BNMF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMF.U[i,k] >= 0.0
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert BNMF.V[j,k] >= 0.0
        
    # Then initialise with expectation values
    init = 'exp'
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    
    assert BNMF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMF.U[i,k] == 1./2.
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert BNMF.V[j,k] == 1./3.
    assert BNMF.tau == 3./1.
    
    
""" Test computing values for alpha, beta, mu, tau. """
I,J,K = 5,3,2
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

lambdaU = 2*numpy.ones((I,K))
lambdaV = 3*numpy.ones((J,K))
alpha, beta = 3, 1
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
init = 'exp' #U=1/2,V=1/3

def test_alpha_s():
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    alpha_s = alpha + 6.
    assert BNMF.alpha_s() == alpha_s

def test_beta_s():
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    beta_s = beta + .5*(12*(2./3.)**2) #U*V.T = [[1/6+1/6,..]]
    assert abs(BNMF.beta_s() - beta_s) < 0.000000000000001
    
def test_tauU():
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    #V^2 = [[1/9,1/9],[1/9,1/9],[1/9,1/9]], sum_j V^2 = [2/9,1/3,2/9,2/9,1/3] (index=i)
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMF.tauU(i,k) == tauU[i,k]
        
def test_muU():
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    muU = 1./tauU * ( 3. * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]) - lambdaU )
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMF.muU(tauU[i,k],i,k) == muU[i,k]
        
def test_tauV():
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    #U^2 = [[1/4,1/4],[1/4,1/4],[1/4,1/4],[1/4,1/4],[1/4,1/4]], sum_i U^2 = [1,1,1] (index=j)
    tauV = 3.*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert BNMF.tauV(j,k) == tauV[j,k]
        
def test_muV():
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauV = 3.*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    muV = 1./tauV * ( 3. * numpy.array([[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)]]) - lambdaV )
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert BNMF.muV(tauV[j,k],j,k) == muV[j,k]
        
#TODO: more complicated test case, check single value for U, V, tau
        
        
# Test two iterations, and that the values have changed in U and V.
# This leads to lots of inf and 0's, so also tests whether we can handle that.
def test_run():
    I,J,K = 10,5,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    init = 'exp' #U=1/2,V=1/3
    
    U_prior = numpy.ones((I,K))/2.
    V_prior = numpy.ones((J,K))/3.
    
    iterations = 15
    
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.initialise(init)
    (Us,Vs,taus) = BNMF.run(iterations)
    
    assert BNMF.all_U.shape == (iterations,I,K)
    assert BNMF.all_V.shape == (iterations,J,K)
    assert BNMF.all_tau.shape == (iterations,)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert Us[0,i,k] != U_prior[i,k]
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert Vs[0,j,k] != V_prior[j,k]
    assert taus[1] != alpha/float(beta)