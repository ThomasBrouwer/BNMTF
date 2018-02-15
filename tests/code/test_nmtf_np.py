"""
Unit tests for the methods in the NMTF class (/code/nmtf_np.py).
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../"
sys.path.append(project_location)

import numpy, math, pytest, itertools, random
from BNMTF.code.models.nmtf_np import NMTF


""" Test the initialisation of Omega """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    K = 0
    L = 0
    with pytest.raises(AssertionError) as error:
        NMTF(R1,M,K,L)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        NMTF(R2,M,K,L)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        NMTF(R3,M,K,L)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Test getting an exception if a row or column is entirely unknown
    R = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    
    with pytest.raises(AssertionError) as error:
        NMTF(R,M1,K,L)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        NMTF(R,M2,K,L)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Test whether we made a copy of R with 1's at unknown values
    I,J = 2,4
    R = [[1,2,0,4],[5,0,7,0]]
    M = [[1,1,0,1],[1,0,1,0]]
    K = 2
    L = 3
    R_excl_unknown = [[1,2,1,4],[5,1,7,1]]
    
    nmtf = NMTF(R,M,K,L)
    assert numpy.array_equal(R,nmtf.R)
    assert numpy.array_equal(M,nmtf.M)
    assert nmtf.I == I
    assert nmtf.J == J
    assert nmtf.K == K
    assert nmtf.L == L
    assert numpy.array_equal(R_excl_unknown,nmtf.R_excl_unknown) 
    
    
""" Test initialisation of F, S, G """   
def test_initialisation():
    I,J = 2,3
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    K = 4
    L = 5
    
    # Init FG ones, S random
    init_FG = 'ones'
    init_S = 'random'
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise(init_S,init_FG)
    
    assert numpy.array_equal(numpy.ones((I,K)),nmtf.F)
    assert numpy.array_equal(numpy.ones((J,L)),nmtf.G)
    for (k,l) in itertools.product(range(0,K),range(0,L)):
        assert nmtf.S[k,l] > 0 and nmtf.S[k,l] < 1
    
    # Init FG random, S ones
    init_FG = 'random'
    init_S = 'ones'
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise(init_S,init_FG)
    
    assert numpy.array_equal(numpy.ones((K,L)),nmtf.S)
    for (i,k) in itertools.product(range(0,I),range(0,K)):
        assert nmtf.F[i,k] > 0 and nmtf.F[i,k] < 1
    for (j,l) in itertools.product(range(0,J),range(0,L)):
        assert nmtf.G[j,k] > 0 and nmtf.G[j,k] < 1    
        
    # Init FG kmeans, S exponential
    init_FG = 'kmeans'
    init_S = 'exponential'
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise(init_S,init_FG)
    
    for (i,k) in itertools.product(range(0,I),range(0,K)):
        assert nmtf.F[i,k] == 0.2 or nmtf.F[i,k] == 1.2
    for (j,l) in itertools.product(range(0,J),range(0,L)):
        assert nmtf.G[j,k] == 0.2 or nmtf.G[j,k] == 1.2   
    for (k,l) in itertools.product(range(0,K),range(0,L)):
        assert nmtf.S[k,l] > 0
    
    
""" Test updates for F, G, S, without dynamic behaviour. """
def test_updates():
    R = numpy.array([[1,2],[3,4]],dtype='f')
    M = numpy.array([[1,1],[0,1]])
    I,J,K,L = 2,2,3,1
    
    F = numpy.array([[1,2,3],[4,5,6]],dtype='f')
    S = numpy.array([[7],[8],[9]],dtype='f')
    G = numpy.array([[10],[11]],dtype='f')
    FSG = numpy.array([[500,550],[1220,1342]],dtype='f')
    FS = numpy.array([[50],[122]],dtype='f')
    SG = numpy.array([[70,77],[80,88],[90,99]],dtype='f')
    
    F_updated = [
        [
            F[0][0] * ( R[0][0]*SG[0][0]/FSG[0][0] + R[0][1]*SG[0][1]/FSG[0][1] ) / ( SG[0][0]+SG[0][1] ),
            F[0][1] * ( R[0][0]*SG[1][0]/FSG[0][0] + R[0][1]*SG[1][1]/FSG[0][1] ) / ( SG[1][0]+SG[1][1] ),
            F[0][2] * ( R[0][0]*SG[2][0]/FSG[0][0] + R[0][1]*SG[2][1]/FSG[0][1] ) / ( SG[2][0]+SG[2][1] )
        ],    
        [
            F[1][0] * R[1][1]/FSG[1][1],
            F[1][1] * R[1][1]/FSG[1][1],
            F[1][2] * R[1][1]/FSG[1][1]
        ]
    ]
    G_updated = [
        [
            G[0][0] * R[0][0]/FSG[0][0]       
        ],
        [
            G[1][0] * ( R[0][1]*FS[0][0]/FSG[0][1] + R[1][1]*FS[1][0]/FSG[1][1] ) / ( FS[0][0]+FS[1][0] )
        ]
    ]
    S_updated = [
        [
            S[0][0] * ( R[0][0]*F[0][0]*G[0][0]/FSG[0][0] + R[0][1]*F[0][0]*G[1][0]/FSG[0][1] + R[1][1]*F[1][0]*G[1][0]/FSG[1][1] ) / ( F[0][0]*G[0][0]+F[0][0]*G[1][0]+F[1][0]*G[1][0] ) 
        ],
        [
            S[1][0] * ( R[0][0]*F[0][1]*G[0][0]/FSG[0][0] + R[0][1]*F[0][1]*G[1][0]/FSG[0][1] + R[1][1]*F[1][1]*G[1][0]/FSG[1][1] ) / ( F[0][1]*G[0][0]+F[0][1]*G[1][0]+F[1][1]*G[1][0] )  
        ],
        [
            S[2][0] * ( R[0][0]*F[0][2]*G[0][0]/FSG[0][0] + R[0][1]*F[0][2]*G[1][0]/FSG[0][1] + R[1][1]*F[1][2]*G[1][0]/FSG[1][1] ) / ( F[0][2]*G[0][0]+F[0][2]*G[1][0]+F[1][2]*G[1][0] )  
        ]
    ]    
    
    nmtf = NMTF(R,M,K,L)
    def reset():
        nmtf.F = numpy.copy(F)
        nmtf.S = numpy.copy(S)
        nmtf.G = numpy.copy(G)
    
    print F[0][0], ( R[0][0]*SG[0][0]/FSG[0][0] + R[0][1]*SG[0][1]/FSG[0][1] ), ( SG[0][0]+SG[0][1] ) 
    print F[1][0], R[1][1]/FSG[1][1]*SG[0][1], SG[0][1]
    
    # Test F
    for k in range(0,K):
        reset()
        nmtf.update_F(k)
        for i in range(0,I):
            print i,k
            assert abs(F_updated[i][k] - nmtf.F[i,k]) < 0.000000001 
            
    # Test G
    for l in range(0,L):
        reset()
        nmtf.update_G(l)
        for j in range(0,J):
            assert abs(G_updated[j][l] - nmtf.G[j,l]) < 0.000000001 
    
    # Test S
    def test_S(k,l):
        reset()
        nmtf.update_S(k,l)
        assert abs(S_updated[k][l] - nmtf.S[k,l]) < 0.00000001
    test_S(0,0)
    test_S(1,0)
    test_S(2,0)
    
    
    
""" Test iterations - whether we get no exception """
def test_run():
    ###### Test updating F, G, S in that order
    # Test whether a single iteration gives the correct behaviour, updating F, G, S in that order
    R = numpy.array([[1,2],[3,4]],dtype='f')
    M = numpy.array([[1,1],[0,1]])
    K = 3
    L = 1
    
    F = numpy.array([[1,2,3],[4,5,6]],dtype='f')
    S = numpy.array([[7],[8],[9]],dtype='f')
    G = numpy.array([[10],[11]],dtype='f')
    #FSG = numpy.array([[500,550],[1220,1342]],dtype='f')
    #FS = numpy.array([[50],[122]],dtype='f')
    #SG = numpy.array([[70,77],[80,88],[90,99]],dtype='f')
    
    nmtf = NMTF(R,M,K,L)
    
    # Check we get an Exception if W, H are undefined
    with pytest.raises(AssertionError) as error:
        nmtf.run(0)
    assert str(error.value) == "F, S and G have not been initialised - please run NMTF.initialise() first."       
    
    nmtf.F = numpy.copy(F)
    nmtf.S = numpy.copy(S)
    nmtf.G = numpy.copy(G) 
    
    nmtf.run(1)
    
    

""" Test divergence calculation """
def test_compute_I_div():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    (I,J,K,L) = (2,2,3,1)
    
    F = numpy.array([[1,2,3],[4,5,6]])
    S = numpy.array([[7],[8],[9]])
    G = numpy.array([[10],[11]])
    #R_predicted = numpy.array([[500,550],[1220,1342]],dtype=float)    
    
    nmtf = NMTF(R,M,K,L)
    nmtf.F = F
    nmtf.S = S
    nmtf.G = G
    
    expected_I_div = sum([
        1.0*math.log(1.0/500.0) - 1.0 + 500.0,
        2.0*math.log(2.0/550.0) - 2.0 + 550.0,
        4.0*math.log(4.0/1342.0) - 4.0 + 1342.0
    ])
    
    nmtf = NMTF(R,M,K,L)
    nmtf.F = F
    nmtf.S = S
    nmtf.G = G
    
    I_div = nmtf.compute_I_div()    
    assert I_div == expected_I_div
    
    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    (I,J,K,L) = (2,2,3,1)
    
    F = numpy.array([[1,2,3],[4,5,6]])
    S = numpy.array([[7],[8],[9]])
    G = numpy.array([[10],[11]])
    #R_predicted = numpy.array([[500,550],[1220,1342]],dtype=float)    
    
    nmtf = NMTF(R,M,K,L)
    nmtf.F = F
    nmtf.S = S
    nmtf.G = G
    performances = nmtf.predict(M)
    
    MSE = ( 499*499 + 548*548 + 1338*1338 ) / 3.0
    R2 = 1 - (499**2 + 548**2 + 1338**2)/((4.0/3.0)**2 + (1.0/3.0)**2 + (5.0/3.0)**2) #mean=7.0/3.0
    #mean_real=7.0/3.0,mean_pred=2392.0/3.0 -> diff_real=[-4.0/3.0,-1.0/3.0,5.0/3.0],diff_pred=[-892.0/3.0,-742.0/3.0,1634.0/3.0]
    Rp = ((-4.0/3.0*-892.0/3.0)+(-1.0/3.0*-742.0/3.0)+(5.0/3.0*1634.0/3.0)) / (math.sqrt((-4.0/3.0)**2+(-1.0/3.0)**2+(5.0/3.0)**2) * math.sqrt((-892.0/3.0)**2+(-742.0/3.0)**2+(1634.0/3.0)**2))
      
    assert performances['MSE'] == MSE
    assert abs(performances['R^2'] - R2) < 0.000000001
    assert abs(performances['Rp'] - Rp) < 0.000000001
       
        
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    (I,J,K,L) = 2, 2, 3, 4
    
    nmtf = NMTF(R,M,K,L)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == nmtf.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == nmtf.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == nmtf.compute_Rp(M_pred,R,R_pred)