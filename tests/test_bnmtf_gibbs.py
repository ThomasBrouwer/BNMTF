"""
Tests for the BNMTF Gibbs sampler.
"""

import numpy, math, pytest, itertools
from BNMTF.code.bnmtf_gibbs import bnmtf_gibbs


""" Test constructor """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array.
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    I,J,K,L = 5,3,1,2
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    alpha, beta = 3, 1    
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R1,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R2,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R3,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Similarly for lambdaF, lambdaS, lambdaG
    I,J,K,L = 2,3,1,2
    R4 = numpy.ones((2,3))
    lambdaF = numpy.ones((2+1,1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R4,M,K,L,priors)
    assert str(error.value) == "Prior matrix lambdaF has the wrong shape: (3, 1) instead of (2, 1)."
    
    lambdaF = numpy.ones((2,1))
    lambdaS = numpy.ones((1+1,2+1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R4,M,K,L,priors)
    assert str(error.value) == "Prior matrix lambdaS has the wrong shape: (2, 3) instead of (1, 2)."
    
    lambdaS = numpy.ones((1,2))
    lambdaG = numpy.ones((3,2+1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R4,M,K,L,priors)
    assert str(error.value) == "Prior matrix lambdaG has the wrong shape: (3, 3) instead of (3, 2)."
    
    # Test getting an exception if a row or column is entirely unknown
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R4,M1,K,L,priors)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        bnmtf_gibbs(R4,M2,K,L,priors)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K = 3,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    M = numpy.ones((I,J))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    BNMTF = bnmtf_gibbs(R5,M,K,L,priors)
    
    assert numpy.array_equal(BNMTF.R,R5)
    assert numpy.array_equal(BNMTF.M,M)
    assert BNMTF.I == I
    assert BNMTF.J == J
    assert BNMTF.K == K
    assert BNMTF.L == L
    assert BNMTF.size_Omega == I*J
    assert BNMTF.alpha == alpha
    assert BNMTF.beta == beta
    assert numpy.array_equal(BNMTF.lambdaF,lambdaF)
    assert numpy.array_equal(BNMTF.lambdaS,lambdaS)
    assert numpy.array_equal(BNMTF.lambdaG,lambdaG)
    
    
""" Test initialing parameters """
def test_initialise():
    I,J,K,L = 5,3,2,4
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    # First do a random initialisation - we can then only check whether values are correctly initialised
    init = 'random'
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    assert BNMTF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.F[i,k] >= 0.0
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.S[k,l] >= 0.0
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.G[j,l] >= 0.0
        
    # Then initialise with expectation values
    init = 'exp'
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    
    assert BNMTF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert BNMTF.F[i,k] >= 1./2.
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert BNMTF.S[k,l] >= 1./3.
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.G[j,l] >= 1./4.
    assert BNMTF.tau == 3./1.
    
    
""" Test computing values for alpha, beta, mu, tau. """
I,J,K,L = 5,3,2,4
R = numpy.ones((I,J))
M = numpy.ones((I,J))
M[0,0], M[2,2], M[3,1] = 0, 0, 0

lambdaF = 2*numpy.ones((I,K))
lambdaS = 3*numpy.ones((K,L))
lambdaG = 5*numpy.ones((J,L))
alpha, beta = 3, 1
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
init = 'exp' #F=1/2,S=1/3,G=1/5

def test_alpha_s():
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    alpha_s = alpha + 6.
    assert BNMTF.alpha_s() == alpha_s

def test_beta_s():
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    beta_s = beta + .5*(12*(11./15.)**2) #F*S = [[1/6+1/6=1/3,..]], F*S*G^T = [[1/15*4=4/15,..]]
    assert abs(BNMTF.beta_s() - beta_s) < 0.00000000000001
    
def test_tauF():
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    # S*G.T = [[4/15]], S*G.T = [[16/225]], sum_j S*G.T = [[32/225,32/225],[48/225,48/225],[32/225,32/225],[32/225,32/225],[48/225,48/225]]
    tauF = 3.*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(BNMTF.tauF(i,k) - tauF[i,k]) < 0.000000000000001
        
#TODO:
def test_muF():
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    #U*V^T - Uik*Vjk = [[1/6,..]], so Rij - Ui * Vj + Uik * Vjk = 5/6
    tauU = 3.*numpy.array([[2./9.,2./9.],[1./3.,1./3.],[2./9.,2./9.],[2./9.,2./9.],[1./3.,1./3.]])
    muU = 1./tauU * ( 3. * numpy.array([[2.*(5./6.)*(1./3.),10./18.],[15./18.,15./18.],[10./18.,10./18.],[10./18.,10./18.],[15./18.,15./18.]]) - lambdaU )
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(BNMTF.muU(tauU[i,k],i,k) - muU[i,k]) < 0.000000000000001
        
def test_tauS():
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    # F outer G = [[1/10]], (F outer G)^2 = [[1/100]], sum (F outer G)^2 = [[12/100]]
    tauS = 3.*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert abs(BNMTF.tauS(k,l) - tauS[k,l]) < 0.000000000000001
    
#TODO:
def test_muS():
    assert False
        
def test_tauG():
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    # F*S = [[1/3]], S*G.T = [[1/9]], sum_i F*S = [[4/9]]
    tauG = 3.*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert BNMTF.tauG(j,l) == tauG[j,l]
    
#TODO:
def test_muG():
    assert False
      
      
""" Test some iterations, and that the values have changed in U and V. """
def test_run():
    I,J,K,L = 10,5,3,2
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    M[0,0], M[2,2], M[3,1] = 0, 0, 0
    
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    init = 'exp' #F=1/2,S=1/3,G=1/4
    
    F_prior = numpy.ones((I,K))/2.
    S_prior = numpy.ones((K,L))/3.
    G_prior = numpy.ones((J,L))/4.
    
    iterations = 15
    
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.initialise(init)
    (Fs,Ss,Gs,taus) = BNMTF.run(iterations)
    
    assert BNMTF.all_F.shape == (iterations,I,K)
    assert BNMTF.all_S.shape == (iterations,K,L)
    assert BNMTF.all_G.shape == (iterations,J,L)
    assert BNMTF.all_tau.shape == (iterations,)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert Fs[0,i,k] != F_prior[i,k]
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert Ss[0,k,l] != S_prior[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert Gs[0,j,l] != G_prior[j,l]
    assert taus[1] != alpha/float(beta)
    
    
#TODO:
""" Test approximating the expectations for F, S, G, tau """
def test_approx_expectation():
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    (I,J,K,L) = (5,3,2,4)
    Fs = [numpy.ones((I,K)) * 3*m**2 for m in range(1,10+1)] 
    Ss = [numpy.ones((K,L)) * 2*m**2 for m in range(1,10+1)]
    Gs = [numpy.ones((J,L)) * 1*m**2 for m in range(1,10+1)] #first is 1's, second is 4's, third is 9's, etc.
    taus = [m**2 for m in range(1,10+1)]
    
    expected_exp_tau = (9.+36.+81.)/3.
    expected_exp_F = numpy.array([[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.],[9.+36.+81.,9.+36.+81.]])
    expected_exp_S = numpy.array([[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)],[(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.),(9.+36.+81.)*(2./3.)]])
    expected_exp_G = numpy.array([[(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)],[(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)],[(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.),(9.+36.+81.)*(1./3.)]])
    
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    BNMTF.all_F = Fs
    BNMTF.all_S = Ss
    BNMTF.all_G = Gs
    BNMTF.all_tau = taus
    (exp_F, exp_S, exp_G, exp_tau) = BNMTF.approx_expectation(burn_in,thinning)
    
    assert expected_exp_tau == exp_tau
    assert numpy.array_equal(expected_exp_F,exp_F)
    assert numpy.array_equal(expected_exp_S,exp_S)
    assert numpy.array_equal(expected_exp_G,exp_G)

    
#TODO:
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    burn_in = 2
    thinning = 3 # so index 2,5,8 -> m=3,m=6,m=9
    (I,J,K) = (5,3,2)
    Us = [numpy.ones((I,K)) * 3*m**2 for m in range(1,10+1)] #first is 1's, second is 4's, third is 9's, etc.
    Vs = [numpy.ones((J,K)) * 2*m**2 for m in range(1,10+1)]
    Us[2][0,0] = 24 #instead of 27 - to ensure we do not get 0 variance in our predictions
    taus = [m**2 for m in range(1,10+1)]
    
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    K = 3
    lambdaU = 2*numpy.ones((I,K))
    lambdaV = 3*numpy.ones((J,K))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }
    
    #expected_exp_U = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    #expected_exp_V = numpy.array([[84.,84.],[84.,84.],[84.,84.]])
    #R_pred = numpy.array([[21084.,21084.,21084.],[ 21168.,21168.,21168.],[21168.,21168.,21168.],[21168.,21168.,21168.],[21168.,21168.,21168.]])
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, P_pred->21084,21168,21168,21168
    MSE = (444408561. + 447872569. + 447660964. + 447618649) / 4.
    R2 = 1. - (444408561. + 447872569. + 447660964. + 447618649) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=21147,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    BNMF = bnmf_gibbs(R,M,K,priors)
    BNMF.all_U = Us
    BNMF.all_V = Vs
    BNMF.all_tau = taus
    performances = BNMF.predict(M_test,burn_in,thinning)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
    
    
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K, L = 2, 2, 3, 4
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    BNMTF = bnmtf_gibbs(R,M,K,L,priors)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == BNMTF.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == BNMTF.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == BNMTF.compute_Rp(M_pred,R,R_pred)
    