"""
Tests for the Iterated Conditional Modes algorithm.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy, math, pytest, itertools
from BNMTF.code.models.nmf_icm import nmf_icm


""" Test constructor """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array.
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    I,J,K = 5,3,1
    lambdaU = numpy.ones((I,K))
    lambdaV = numpy.ones((J,K))
    alpha, beta = 3, 1    
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    with pytest.raises(AssertionError) as error:
        nmf_icm(R1,M,K,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        nmf_icm(R2,M,K,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        nmf_icm(R3,M,K,priors)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Similarly for lambdaU, lambdaV
    R4 = numpy.ones((2,3))
    lambdaU = numpy.ones((2+1,1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M,K,priors)
    assert str(error.value) == "Prior matrix lambdaU has the wrong shape: (3, 1) instead of (2, 1)."
    
    lambdaU = numpy.ones((2,1))
    lambdaV = numpy.ones((3+1,1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M,K,priors)
    assert str(error.value) == "Prior matrix lambdaV has the wrong shape: (4, 1) instead of (3, 1)."
    
    # Test getting an exception if a row or column is entirely unknown
    lambdaU = numpy.ones((2,1))
    lambdaV = numpy.ones((3,1))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M1,K,priors)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        nmf_icm(R4,M2,K,priors)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaU = numpy.ones((I,K))
    lambdaV = numpy.ones((J,K))
    M = numpy.ones((I,J))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    NMF = nmf_icm(R5,M,K,priors)
    
    assert numpy.array_equal(NMF.R,R5)
    assert numpy.array_equal(NMF.M,M)
    assert NMF.I == I
    assert NMF.J == J
    assert NMF.K == K
    assert NMF.size_Omega == I*J
    assert NMF.alpha == alpha
    assert NMF.beta == beta
    assert numpy.array_equal(NMF.lambdaU,lambdaU)
    assert numpy.array_equal(NMF.lambdaV,lambdaV)
    
    # And when lambdaU and lambdaV are integers    
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaU = 3.
    lambdaV = 4.
    M = numpy.ones((I,J))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    NMF = nmf_icm(R5,M,K,priors)
    
    assert numpy.array_equal(NMF.R,R5)
    assert numpy.array_equal(NMF.M,M)
    assert NMF.I == I
    assert NMF.J == J
    assert NMF.K == K
    assert NMF.size_Omega == I*J
    assert NMF.alpha == alpha
    assert NMF.beta == beta
    assert numpy.array_equal(NMF.lambdaU,lambdaU*numpy.ones((I,K)))
    assert numpy.array_equal(NMF.lambdaV,lambdaV*numpy.ones((J,K)))
    
    
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
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    
    assert NMF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMF.U[i,k] >= 0.0
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert NMF.V[j,k] >= 0.0
    #assert NMF.tau == 3./1.
        
    # Then initialise with expectation values
    init = 'exp'
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    
    assert NMF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMF.U[i,k] == 1./2.
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert NMF.V[j,k] == 1./3.
    #assert NMF.tau == 3./1.
    
    
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
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    alpha_s = alpha + 6.
    assert NMF.alpha_s() == alpha_s

def test_beta_s():
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    beta_s = beta + .5*(12*(2./3.)**2) #U*V.T = [[1/6+1/6,..]]
    assert abs(NMF.beta_s() - beta_s) < 0.000000000000001
    
def test_tauU():
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    NMF.tau = 3.
    #V^2 = [[1/9,1/9],[1/9,1/9],[1/9,1/9]], sum_j V^2 = [2/9,1/3,2/9,2/9,1/3] (index=i)
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMF.tauU(k)[i] == tauU[i,k]
        
def test_muU():
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    NMF.tau = 3.
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    muU = 1./tauU * ( 3. * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]) - lambdaU )
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(NMF.muU(tauU[:,k],k)[i] - muU[i,k]) < 0.000000000000001
        
def test_tauV():
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    NMF.tau = 3.
    #U^2 = [[1/4,1/4],[1/4,1/4],[1/4,1/4],[1/4,1/4],[1/4,1/4]], sum_i U^2 = [1,1,1] (index=j)
    tauV = 3.*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert NMF.tauV(k)[j] == tauV[j,k]
        
def test_muV():
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    NMF.tau = 3.
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauV = 3.*numpy.array([[1.,1.],[1.,1.],[1.,1.]])
    muV = 1./tauV * ( 3. * numpy.array([[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)],[4.*(5./6.)*(1./2.),4.*(5./6.)*(1./2.)]]) - lambdaV )
    for j,k in itertools.product(xrange(0,J),xrange(0,K)):
        assert NMF.muV(tauV[:,k],k)[j] == muV[j,k]
        
        
""" Test some iterations, and that the values have changed in U and V. """
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
    
    iterations = 15
    
    NMF = nmf_icm(R,M,K,priors)
    NMF.initialise(init)
    NMF.run(iterations)
    
    assert NMF.all_tau.shape == (iterations,)
    assert NMF.all_tau[1] != alpha/float(beta)

    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    (I,J,K) = (5,3,2)
    U = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    V = numpy.array([[84.,84.],[84.,84.],[84.,84.]])
    taus = [m**2 for m in range(1,10+1)]
    #R_pred = numpy.array([[21084.,21084.,21084.],[ 21168.,21168.,21168.],[21168.,21168.,21168.],[21168.,21168.,21168.],[21168.,21168.,21168.]])
    
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, P_pred->21084,21168,21168,21168
    MSE = (444408561. + 447872569. + 447660964. + 447618649) / 4.
    R2 = 1. - (444408561. + 447872569. + 447660964. + 447618649) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=21147,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    NMF = nmf_icm(R,M,K,priors)
    NMF.U = U
    NMF.V = V
    NMF.all_tau = taus
    performances = NMF.predict(M_test)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
    
    
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K = 2, 2, 3
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    BNMF = nmf_icm(R,M,K,priors)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == BNMF.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == BNMF.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == BNMF.compute_Rp(M_pred,R,R_pred)
    
    
""" Test the model quality measures. """
def test_log_likelihood():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K = 2, 2, 3
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    BNMF = nmf_icm(R,M,K,priors)
    BNMF.U = numpy.ones((I,K))
    BNMF.V = 2*numpy.ones((J,K))
    BNMF.tau = 3.
    # expU*expV.T = [[6.]]
    
    log_likelihood = 3./2.*(math.log(3.)-math.log(2*math.pi)) - 3./2. * (5**2 + 4**2 + 2**2)
    AIC = -2*log_likelihood + 2*(2*3+2*3)
    BIC = -2*log_likelihood + (2*3+2*3)*math.log(3)
    MSE = (5**2+4**2+2**2)/3.
    
    assert log_likelihood == BNMF.quality('loglikelihood')
    assert AIC == BNMF.quality('AIC')
    assert BIC == BNMF.quality('BIC')
    assert MSE == BNMF.quality('MSE')
    with pytest.raises(AssertionError) as error:
        BNMF.quality('FAIL')
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."