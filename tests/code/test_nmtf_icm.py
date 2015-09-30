"""
Tests for the NMTF Iterated Conditional Modes algorithm.
"""

import numpy, math, pytest, itertools
from BNMTF.code.nmtf_icm import nmtf_icm


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
        nmtf_icm(R1,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R2,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R3,M,K,L,priors)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Similarly for lambdaF, lambdaS, lambdaG
    I,J,K,L = 2,3,1,2
    R4 = numpy.ones((2,3))
    lambdaF = numpy.ones((2+1,1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R4,M,K,L,priors)
    assert str(error.value) == "Prior matrix lambdaF has the wrong shape: (3, 1) instead of (2, 1)."
    
    lambdaF = numpy.ones((2,1))
    lambdaS = numpy.ones((1+1,2+1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R4,M,K,L,priors)
    assert str(error.value) == "Prior matrix lambdaS has the wrong shape: (2, 3) instead of (1, 2)."
    
    lambdaS = numpy.ones((1,2))
    lambdaG = numpy.ones((3,2+1))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R4,M,K,L,priors)
    assert str(error.value) == "Prior matrix lambdaG has the wrong shape: (3, 3) instead of (3, 2)."
    
    # Test getting an exception if a row or column is entirely unknown
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R4,M1,K,L,priors)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        nmtf_icm(R4,M2,K,L,priors)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Finally, a successful case
    I,J,K,L = 3,2,2,2
    R5 = 2*numpy.ones((I,J))
    lambdaF = numpy.ones((I,K))
    lambdaS = numpy.ones((K,L))
    lambdaG = numpy.ones((J,L))
    M = numpy.ones((I,J))
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    NMTF = nmtf_icm(R5,M,K,L,priors)
    
    assert numpy.array_equal(NMTF.R,R5)
    assert numpy.array_equal(NMTF.M,M)
    assert NMTF.I == I
    assert NMTF.J == J
    assert NMTF.K == K
    assert NMTF.L == L
    assert NMTF.size_Omega == I*J
    assert NMTF.alpha == alpha
    assert NMTF.beta == beta
    assert numpy.array_equal(NMTF.lambdaF,lambdaF)
    assert numpy.array_equal(NMTF.lambdaS,lambdaS)
    assert numpy.array_equal(NMTF.lambdaG,lambdaG)
    
    
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
    init_S = 'random'
    init_FG = 'random'
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    
    assert NMTF.tau >= 0.0
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMTF.F[i,k] >= 0.0
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert NMTF.S[k,l] >= 0.0
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert NMTF.G[j,l] >= 0.0
        
    # Initialisation of S using random draws from prior
    init_S, init_FG = 'random', 'exp'
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMTF.F[i,k] == 1./lambdaF[i,k]
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert NMTF.S[k,l] != 1./lambdaS[k,l] # test whether we overwrote the expectation
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert NMTF.G[j,l] == 1./lambdaG[j,l]
    
    # Initialisation of F and G using random draws from prior
    init_S, init_FG = 'exp', 'random'
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMTF.F[i,k] != 1./lambdaF[i,k] # test whether we overwrote the expectation
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert NMTF.S[k,l] == 1./lambdaS[k,l]
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert NMTF.G[j,l] != 1./lambdaG[j,l] # test whether we overwrote the expectation
        
    # Initialisation of F and G using Kmeans
    init_S, init_FG = 'exp', 'kmeans'
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert NMTF.F[i,k] == 0.2 or NMTF.F[i,k] == 1.2
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert NMTF.G[j,l] == 0.2 or NMTF.G[j,l] == 1.2
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert NMTF.S[k,l] == 1./lambdaS[k,l]
    
    
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
init_S, init_FG = 'exp', 'exp'
# F = 1/2, S = 1/3, G = 1/5
# R - FSG.T = [[1]] - [[4/15]] = [[11/15]]

def test_alpha_s():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    alpha_s = alpha + 6.
    assert NMTF.alpha_s() == alpha_s

def test_beta_s():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    beta_s = beta + .5*(12*(11./15.)**2) #F*S = [[1/6+1/6=1/3,..]], F*S*G^T = [[1/15*4=4/15,..]]
    assert abs(NMTF.beta_s() - beta_s) < 0.00000000000001
    
def test_tauF():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    # S*G.T = [[4/15]], (S*G.T)^2 = [[16/225]], sum_j S*G.T = [[32/225,32/225],[48/225,48/225],[32/225,32/225],[32/225,32/225],[48/225,48/225]]
    tauF = 3.*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(NMTF.tauF(k)[i] - tauF[i,k]) < 0.000000000000001
        
def test_muF():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    tauF = 3.*numpy.array([[32./225.,32./225.],[48./225.,48./225.],[32./225.,32./225.],[32./225.,32./225.],[48./225.,48./225.]])
    # Rij - Fi*S*Gj + Fik(Sk*Gj) = 11/15 + 1/2 * 4/15 = 13/15
    # (Rij - Fi*S*Gj + Fik(Sk*Gj)) * (Sk*Gj) = 13/15 * 4/15 = 52/225
    muF = 1./tauF * ( 3. * numpy.array([[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)],[2*(52./225.),2*(52./225.)],[2*(52./225.),2*(52./225.)],[3*(52./225.),3*(52./225.)]]) - lambdaF )
    for i,k in itertools.product(xrange(0,I),xrange(0,K)):
        assert abs(NMTF.muF(tauF[:,k],k)[i] - muF[i,k]) < 0.000000000000001
        
def test_tauS():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    # F outer G = [[1/10]], (F outer G)^2 = [[1/100]], sum (F outer G)^2 = [[12/100]]
    tauS = 3.*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert abs(NMTF.tauS(k,l) - tauS[k,l]) < 0.000000000000001
    
def test_muS():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    tauS = 3.*numpy.array([[3./25.,3./25.,3./25.,3./25.],[3./25.,3./25.,3./25.,3./25.]])
    # Rij - Fi*S*Gj + Fik*Skl*Gjk = 11/15 + 1/2*1/3*1/5 = 23/30
    # (Rij - Fi*S*Gj + Fik*Skl*Gjk) * Fik*Gjk = 23/30 * 1/10 = 23/300
    muS = 1./tauS * ( 3. * numpy.array([[12*23./300.,12*23./300.,12*23./300.,12*23./300.],[12*23./300.,12*23./300.,12*23./300.,12*23./300.]]) - lambdaS )
    for k,l in itertools.product(xrange(0,K),xrange(0,L)):
        assert abs(NMTF.muS(tauS[k,l],k,l) - muS[k,l]) < 0.000000000000001
        
def test_tauG():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    # F*S = [[1/3]], (F*S)^2 = [[1/9]], sum_i F*S = [[4/9]]
    tauG = 3.*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert NMTF.tauG(l)[j] == tauG[j,l]
    
def test_muG():
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init_S,init_FG)
    NMTF.tau = 3.
    tauG = 3.*numpy.array([[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.],[4./9.,4./9.,4./9.,4./9.]])
    # Rij - Fi*S*Gj + Gjl*(Fi*Sl)) = 11/15 + 1/5 * 1/3 = 12/15 = 4/5
    # (Rij - Fi*S*Gj + Gjl*(Fi*Sl)) * (Fi*Sl) = 4/5 * 1/3 = 4/15
    muG = 1./tauG * ( 3. * numpy.array([[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.],[4.*4./15.,4.*4./15.,4.*4./15.,4.*4./15.]]) - lambdaG )
    for j,l in itertools.product(xrange(0,J),xrange(0,L)):
        assert abs(NMTF.muG(tauG[:,l],l)[j] - muG[j,l]) < 0.000000000000001
      
      
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
    
    iterations = 15
    
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.initialise(init)
    NMTF.run(iterations)
    
    assert NMTF.all_tau.shape == (iterations,)
    assert NMTF.all_tau[1] != alpha/float(beta)
    
    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    (I,J,K,L) = (5,3,2,4)
    F = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    S = numpy.array([[84.,84.,84.,84.],[84.,84.,84.,84.]])
    G = numpy.array([[42.,42.,42.,42.],[42.,42.,42.,42.],[42.,42.,42.,42.]])
    taus = [m**2 for m in range(1,10+1)]
    #R_pred = numpy.array([[ 3542112.,  3542112.,  3542112.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.],[ 3556224.,  3556224.,  3556224.]])
     
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 5*numpy.ones((J,L))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->3542112,3556224,3556224,3556224
    MSE = ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / 4.
    R2 = 1. - ((3.-3542112.)**2 + (5.-3556224.)**2 + (10.-3556224.)**2 + (11.-3556224.)**2) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=3552696,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.F = F
    NMTF.S = S
    NMTF.G = G
    NMTF.all_tau = taus
    performances = NMTF.predict(M_test)
    
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
    
    NMTF = nmtf_icm(R,M,K,L,priors)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == NMTF.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == NMTF.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == NMTF.compute_Rp(M_pred,R,R_pred)
    
    
""" Test the model quality measures. """
def test_log_likelihood():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    I, J, K, L = 2, 2, 3, 4
    lambdaF = 2*numpy.ones((I,K))
    lambdaS = 3*numpy.ones((K,L))
    lambdaG = 4*numpy.ones((J,L))
    alpha, beta = 3, 1
    priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }
    
    NMTF = nmtf_icm(R,M,K,L,priors)
    NMTF.F = numpy.ones((I,K))
    NMTF.S = 2*numpy.ones((K,L))
    NMTF.G = 3*numpy.ones((J,L))
    NMTF.tau = 3.
    # expU*expV.T = [[72.]]
    
    log_likelihood = 3./2.*(math.log(3)-math.log(2*math.pi)) - 3./2. * (71**2 + 70**2 + 68**2)
    AIC = -2*log_likelihood + 2*(2*3+3*4+2*4)
    BIC = -2*log_likelihood + (2*3+3*4+2*4)*math.log(3)
    MSE = (71**2+70**2+68**2)/3.
    
    assert log_likelihood == NMTF.quality('loglikelihood')
    assert AIC == NMTF.quality('AIC')
    assert BIC == NMTF.quality('BIC')
    assert MSE == NMTF.quality('MSE')
    with pytest.raises(AssertionError) as error:
        NMTF.quality('FAIL')
    assert str(error.value) == "Unrecognised metric for model quality: FAIL."