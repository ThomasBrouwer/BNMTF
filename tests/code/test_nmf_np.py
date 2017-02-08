"""
Unit tests for the methods in the NMF class (/code/nmf_np.py).
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy, math, pytest, itertools
from BNMTF.code.models.nmf_np import NMF


""" Test the initialisation of Omega """
def test_init():
    # Test getting an exception when R and M are different sizes, and when R is not a 2D array
    R1 = numpy.ones(3)
    M = numpy.ones((2,3))
    K = 0
    with pytest.raises(AssertionError) as error:
        NMF(R1,M,K)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 1-dimensional."
    
    R2 = numpy.ones((4,3,2))
    with pytest.raises(AssertionError) as error:
        NMF(R2,M,K)
    assert str(error.value) == "Input matrix R is not a two-dimensional array, but instead 3-dimensional."
    
    R3 = numpy.ones((3,2))
    with pytest.raises(AssertionError) as error:
        NMF(R3,M,K)
    assert str(error.value) == "Input matrix R is not of the same size as the indicator matrix M: (3, 2) and (2, 3) respectively."
    
    # Test getting an exception if a row or column is entirely unknown
    R = numpy.ones((2,3))
    M1 = [[1,1,1],[0,0,0]]
    M2 = [[1,1,0],[1,0,0]]
    
    with pytest.raises(AssertionError) as error:
        NMF(R,M1,K)
    assert str(error.value) == "Fully unobserved row in R, row 1."
    with pytest.raises(AssertionError) as error:
        NMF(R,M2,K)
    assert str(error.value) == "Fully unobserved column in R, column 2."
    
    # Test whether we made a copy of R with 1's at unknown values
    I,J = 2,4
    R = [[1,2,0,4],[5,0,7,0]]
    M = [[1,1,0,1],[1,0,1,0]]
    K = 2
    R_excl_unknown = [[1,2,1,4],[5,1,7,1]]
    
    nmf = NMF(R,M,K)
    assert numpy.array_equal(R,nmf.R)
    assert numpy.array_equal(M,nmf.M)
    assert nmf.I == I
    assert nmf.J == J
    assert nmf.K == K
    assert numpy.array_equal(R_excl_unknown,nmf.R_excl_unknown)
    
        
    
""" Test initialisation of U, V """   
def test_initialisation():
    I,J = 2,3
    R = numpy.ones((I,J))
    M = numpy.ones((I,J))
    K = 4
    
    # Init ones
    init_UV = 'ones'
    nmf = NMF(R,M,K)
    nmf.initialise(init_UV)
    
    assert numpy.array_equal(numpy.ones((2,4)),nmf.U)
    assert numpy.array_equal(numpy.ones((3,4)),nmf.V)
    
    # Init random
    init_UV = 'random'
    nmf = NMF(R,M,K)
    nmf.initialise(init_UV)
    
    for (i,k) in itertools.product(range(0,I),range(0,K)):
        assert nmf.U[i,k] > 0 and nmf.U[i,k] < 1
    for (j,k) in itertools.product(range(0,J),range(0,K)):
        assert nmf.V[j,k] > 0 and nmf.V[j,k] < 1
    
    
""" Test updates for Uik, Vjk """
def test_update():
    I,J = 2,4
    R = [[1,2,0,4],[5,0,7,0]]
    M = [[1,1,0,1],[1,0,1,0]]
    K = 2
    
    U = numpy.array([[1,2],[3,4]],dtype='f') #2x2
    V = numpy.array([[5,6],[7,8],[9,10],[11,12]],dtype='f') #4x2
    
    new_U = [[
        1 * ( 1*5/17.0 + 2*7/23.0 + 4*11/35.0 ) / float( 5+7+11 ),
        2 * ( 1*6/17.0 + 2*8/23.0 + 4*12/35.0 ) / float( 6+8+12 )
    ],[
        3 * ( 5*5/39.0 + 7*9/67.0 ) / float( 5+9 ),
        4 * ( 5*6/39.0 + 7*10/67.0 ) / float( 6+10 )
    ]]
    
    new_V = [[
        5 * ( 1*1/17.0 + 3*5/39.0 ) / float( 1+3 ),
        6 * ( 2*1/17.0 + 4*5/39.0 ) / float( 2+4 )
    ],[
        7 * ( 1*2/23.0 ) / float( 1 ),
        8 * ( 2*2/23.0 ) / float( 2 )
    ],[
        9 * ( 3*7/67.0 ) / float( 3 ),
        10 * ( 4*7/67.0 ) / float( 4 )
    ],[
        11 * ( 1*4/35.0 ) / float( 1 ),
        12 * ( 2*4/35.0 ) / float( 2 )
    ]]
    
    nmf = NMF(R,M,K)
    def reset():
        nmf.U = numpy.copy(U)
        nmf.V = numpy.copy(V)
    
    for k in range(0,K):
        reset()
        nmf.update_U(k)      
        
        for i in range(0,I):
            assert abs(new_U[i][k] - nmf.U[i,k]) < 0.00001
        
    for k in range(0,K):
        reset()
        nmf.update_V(k)
        for j in range(0,J):
            assert abs(new_V[j][k] - nmf.V[j,k]) < 0.00001
    
    # Also if I = J
    I,J,K = 2,2,3
    R = [[1,2],[3,4]]
    M = [[1,1],[0,1]]
    
    U = numpy.array([[1,2,3],[4,5,6]],dtype='f') #2x3
    V = numpy.array([[7,8,9],[10,11,12]],dtype='f') #2x3
    R_pred = numpy.array([[50,68],[122,167]],dtype='f') #2x2
    
    nmf = NMF(R,M,K)
    def reset_2():
        nmf.U = numpy.copy(U)
        nmf.V = numpy.copy(V)
        
    for k in range(0,K):
        reset_2()
        nmf.update_U(k)    
        for i in range(0,I):
            new_Uik = U[i][k] * sum( [V[j][k] * R[i][j] / R_pred[i,j] for j in range(0,J) if M[i][j] ]) \
                              / sum( [V[j][k] for j in range(0,J) if M[i][j] ])
            assert abs(new_Uik - nmf.U[i,k]) < 0.00001
            
    for k in range(0,K):
        reset_2()
        nmf.update_V(k)    
        for j in range(0,J):
            new_Vjk = V[j][k] * sum( [U[i][k] * R[i][j] / R_pred[i,j] for i in range(0,I) if M[i][j] ]) \
                              / sum( [U[i][k] for i in range(0,I) if M[i][j] ])
            assert abs(new_Vjk - nmf.V[j,k]) < 0.00001


""" Test iterations - whether we get no exception """
def test_run():
    # Data generated from W = [[1,2],[3,4]], H = [[4,3],[2,1]]
    R = [[8,5],[20,13]]
    M = [[1,1],[1,0]]
    K = 2
    
    U = numpy.array([[10,9],[8,7]],dtype='f') #2x2
    V = numpy.array([[6,4],[5,3]],dtype='f') #2x2 
    
    nmf = NMF(R,M,K)
    
    # Check we get an Exception if W, H are undefined
    with pytest.raises(AssertionError) as error:
        nmf.run(0)
    assert str(error.value) == "U and V have not been initialised - please run NMF.initialise() first."    
    
    # Then check for 1 iteration whether the updates work - heck just the first entry of U
    nmf.U = U
    nmf.V = V
    nmf.run(1)
    
    U_00 = 10*(6*8/96.0+5*5/77.0)/(5.0+6.0) #0.74970484061
    assert abs(U_00 - nmf.U[0][0]) < 0.000001
    

""" Test divergence calculation """
def test_compute_I_div():
    R = [[1,2,0,4],[5,0,7,0]]
    M = [[1,1,0,1],[1,0,1,0]]
    K = 2
    
    U = numpy.array([[1,2],[3,4]],dtype='f') #2x2
    V = numpy.array([[5,7,9,11],[6,8,10,12]],dtype='f').T #4x2 
    #R_pred = [[17,23,29,35],[39,53,67,81]]
    
    expected_I_div = sum([
        1.0*math.log(1.0/17.0) - 1.0 + 17.0,
        2.0*math.log(2.0/23.0) - 2.0 + 23.0,
        4.0*math.log(4.0/35.0) - 4.0 + 35.0,
        5.0*math.log(5.0/39.0) - 5.0 + 39.0,
        7.0*math.log(7.0/67.0) - 7.0 + 67.0
    ])
    
    nmf = NMF(R,M,K)
    nmf.U = U
    nmf.V = V   
    
    I_div = nmf.compute_I_div()    
    assert abs(I_div - expected_I_div) < 0.0000001
    
    
""" Test computing the performance of the predictions using the expectations """
def test_predict():
    (I,J,K) = (5,3,2)
    R = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],dtype=float)
    M = numpy.ones((I,J))
    K = 3
    
    U = numpy.array([[125.,126.],[126.,126.],[126.,126.],[126.,126.],[126.,126.]])
    V = numpy.array([[84.,84.],[84.,84.],[84.,84.]])
    
    M_test = numpy.array([[0,0,1],[0,1,0],[0,0,0],[1,1,0],[0,0,0]]) #R->3,5,10,11, R_pred->21084,21168,21168,21168
    MSE = (444408561. + 447872569. + 447660964. + 447618649) / 4.
    R2 = 1. - (444408561. + 447872569. + 447660964. + 447618649) / (4.25**2+2.25**2+2.75**2+3.75**2) #mean=7.25
    Rp = 357. / ( math.sqrt(44.75) * math.sqrt(5292.) ) #mean=7.25,var=44.75, mean_pred=21147,var_pred=5292, corr=(-4.25*-63 + -2.25*21 + 2.75*21 + 3.75*21)
    
    nmf = NMF(R,M,K)
    nmf.U = U
    nmf.V = V
    performances = nmf.predict(M_test)
    
    assert performances['MSE'] == MSE
    assert performances['R^2'] == R2
    assert performances['Rp'] == Rp
       
        
""" Test the evaluation measures MSE, R^2, Rp """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    (I,J,K) = 2, 2, 3
    
    nmf = NMF(R,M,K)
    
    R_pred = numpy.array([[500,550],[1220,1342]],dtype=float)
    M_pred = numpy.array([[0,0],[1,1]])
    
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    Rp_pred = 61. / ( math.sqrt(.5) * math.sqrt(7442.) ) #mean=3.5,var=0.5,mean_pred=1281,var_pred=7442,cov=61
    
    assert MSE_pred == nmf.compute_MSE(M_pred,R,R_pred)
    assert R2_pred == nmf.compute_R2(M_pred,R,R_pred)
    assert Rp_pred == nmf.compute_Rp(M_pred,R,R_pred)