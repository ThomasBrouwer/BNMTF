"""
Unit tests for the methods in the NMTF class (/code/nmtf.py).
"""

import numpy, math, pytest, itertools, random
from nmtf_i_div.code.nmtf import NMTF


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
    
    # Test completely observed case
    R = numpy.ones((2,3))
    M = numpy.ones((2,3))
    
    Omega = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    OmegaI = [[0,1,2],[0,1,2]]
    OmegaJ = [[0,1],[0,1],[0,1]]    

    nmtf = NMTF(R,M,K,L)
    assert numpy.array_equal(Omega,nmtf.Omega)    
    assert numpy.array_equal(OmegaI,nmtf.OmegaI)    
    assert numpy.array_equal(OmegaJ,nmtf.OmegaJ)    
    
    # Test partially observed case
    M = [[1,0,1],[0,1,1]]
    
    Omega = [(0,0),(0,2),(1,1),(1,2)]
    OmegaI = [[0,2],[1,2]]
    OmegaJ = [[0],[1],[0,1]] 
    
    nmtf = NMTF(R,M,K,L)
    assert numpy.array_equal(Omega,nmtf.Omega)    
    assert numpy.array_equal(OmegaI,nmtf.OmegaI)    
    assert numpy.array_equal(OmegaJ,nmtf.OmegaJ)  
    
    
    
""" Test initialisation of F, S, G """   
def test_initialisation():  
    # Test non-Kmeans initialisation
    R = numpy.ones((2,3))
    M = numpy.ones((2,3))
    K = 4
    L = 5
    
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise()
    assert numpy.array_equal(numpy.ones((2,4)),nmtf.F)
    assert numpy.array_equal(numpy.ones((3,5)),nmtf.G)  
    assert numpy.array_equal(numpy.ones((4,5)),nmtf.S)   
    
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise(Kmeans=False)
    assert numpy.array_equal(numpy.ones((2,4)),nmtf.F)
    assert numpy.array_equal(numpy.ones((3,5)),nmtf.G)  
    assert numpy.array_equal(numpy.ones((4,5)),nmtf.S)  
    
    # Test Kmeans initialisation
    random.seed(0)    
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise(Kmeans=True,S_random=False)
    expected_F = [[1.2,0.2,0.2,0.2],[1.2,0.2,0.2,0.2]]
    expected_G = [[1.2,0.2,0.2,0.2,0.2],[1.2,0.2,0.2,0.2,0.2],[1.2,0.2,0.2,0.2,0.2]]
    assert numpy.array_equal(expected_F,nmtf.F)
    assert numpy.array_equal(expected_G,nmtf.G) 
    assert numpy.array_equal(numpy.ones((4,5)),nmtf.S) 
    
    # And if we let S be initialised randomly
    R = [[1,2,3],[1,2,4],[5,6,10]]   # rows 1 and 2 should be clustered together, and columns 1 and 2
    M = numpy.ones((3,3))
    K = 2
    L = 2
    random.seed(0)   
    numpy.random.seed(0)
    nmtf = NMTF(R,M,K,L)
    nmtf.initialise(Kmeans=True)
    expected_F = [[0.2,1.2],[0.2,1.2],[1.2,0.2]]
    expected_G = [[1.2,0.2],[1.2,0.2],[0.2,1.2]]
    numpy.random.seed(0)
    expected_S = numpy.random.rand(2,2) 
    
    assert numpy.array_equal(expected_F,nmtf.F)
    assert numpy.array_equal(expected_G,nmtf.G) 
    assert numpy.array_equal(expected_S,nmtf.S) 
    
    
    
""" Test computing the prediction performance on the test set """
def test_predict():
    R = numpy.array([[1,2],[3,4]],dtype='f')
    M = [[0,1],[1,0]]
    R_pred = numpy.array([[10,9],[8,7]],dtype='f')
    M_test = [[1,0],[0,1]]
    expected_performances = {
        'MSE' : (81+9)/2.0,
        'R^2' : 1-(81+9)/(1.5**2+1.5**2), #mean=2.5
        'Rp' : (-1.5*1.5+1.5*-1.5)/float(math.sqrt(1.5**2+1.5**2)*math.sqrt(1.5**2+1.5**2)) #mean_real=2.5,real_pred=8.5
    }
    
    nmtf = NMTF(R,M,K=2,L=2)
    nmtf.R_pred = R_pred
    performances = nmtf.predict(R,M_test)
    assert expected_performances == performances
        
    
    
""" Test updates for F, G, S, without dynamic behaviour. """
def test_updates():
    R = numpy.array([[1,2],[3,4]],dtype='f')
    M = numpy.array([[1,1],[0,1]])
    K = 3
    L = 1
    
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
        nmtf.SG = numpy.dot(nmtf.S,nmtf.G.T)
        nmtf.FS = numpy.dot(nmtf.F,nmtf.S)
    
    # Test F
    def test_F(i,k):
        reset()
        nmtf.update_F(i,k)
        assert abs(F_updated[i][k] - nmtf.F[i][k]) < 0.000000001 #ignore rounding errors
    test_F(0,0)
    test_F(0,1)
    test_F(0,2)
    test_F(1,0)    
    test_F(1,1)   
    test_F(1,2)
    
    # Test G
    def test_G(j,l):
        reset()
        nmtf.update_G(j,l)
        assert G_updated[j][l] == nmtf.G[j][l]
    test_G(0,0)
    test_G(1,0)
    
    # Test S
    def test_S(k,l):
        reset()
        nmtf.update_S(k,l)
        assert abs(S_updated[k][l] - nmtf.S[k][l]) < 0.00000001 #ignore rounding errors
    test_S(0,0)
    test_S(1,0)
    test_S(2,0)
    
    
    
""" Test iterations """
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
        nmtf.run(0,0,0)
    assert str(error.value) == "F, S and G have not been initialised - please run NMTF.initialise() first."       
    
    nmtf.F = numpy.copy(F)
    nmtf.S = numpy.copy(S)
    nmtf.G = numpy.copy(G) 
    
    nmtf.run(1,0,0,False)
    
    # Updates to F
    # Before updates, R_pred = [[500,550],[1220,1342]]
    # After updating F_00, R_pred_0 = [430.199999999,473.219999999]
    # After updating F_01, R_pred_0 = [270.731314338,297.804445772]
    # After updating F_10, R_pred_1 = [940.83457526,1034.91803279]
    # After updating F_11, R_pred_1 = [542.380591417,596.618650559]
    F_updated = [[
        1.0 * ( 1.0*70.0/500 + 2.0*77.0/550 ) / ( 70.0+77.0 ),                      #0.00285714285
        2.0 * ( 1.0*80.0/430.199999999 + 2.0*88.0/473.219999999 ) / ( 80.0+88.0 ),  #0.00664142923
        3.0 * ( 1.0*90.0/270.731314338 + 2.0*99.0/297.804445772 ) / ( 90.0+99.0 )   #0.0158301388
    ],[
        4.0 * 4.0 / 1342.0,         #0.01192250372
        5.0 * 4.0 / 1034.91803279,  #0.01932520196
        6.0 * 4.0 / 596.618650559   #0.04022670088
    ]]
    
    for i,k in itertools.product(range(0,2),range(0,3)):
        assert abs(F_updated[i][k] - nmtf.F[i][k]) < 0.0000000000001
    
    # Updates to G
    # Before updates, R_pred = [[2.1560268299,2.37162951289],[6.0009944964,6.60109394604]]
    # FS = [[0.21560268299],[0.60009944964]]
    # After updating G_00, R_pred_(.)_0 = [1,2.78335798664]
    G_updated = [[
        10.0 * 1.0 / 2.1560268299   #4.63816120529  
    ],[
        11.0 * ( 2.0*0.21560268299/2.37162951289 + 4.0*0.60009944964/6.60109394604 ) / ( 0.21560268299+0.60009944964 )  #7.35562622676
    ]]
    
    for j,l in itertools.product(range(0,2),range(0,1)):
        assert abs(G_updated[j][l] - nmtf.G[j][l]) < 0.00000001
    
    # Updates to S
    # Before updates, R_pred = [[1,1.58589274956],[2.78335798664,4.41410725044]]
    # After updating S_00, R_pred = [[0.99791635,1.5825883],[2.77466317,4.40031821]]
    # After updating S_10, R_pred = [[0.9979354,1.58261851],[2.7747186,4.40040611]]
    S_updated = [
        [
            7.0 * ( 1.0*0.00285714285*4.63816120529/1.0 + 2.0*0.00285714285*7.35562622676/1.58589274956 + 4.0*0.01192250372*7.35562622676/4.41410725044 ) / ( 0.00285714285*4.63816120529+0.00285714285*7.35562622676+0.01192250372*7.35562622676 ) 
        ], #6.84276577856
        [
            8.0 * ( 1.0*0.00664142923*4.63816120529/0.99791635 + 2.0*0.00664142923*7.35562622676/1.5825883 + 4.0*0.01932520196*7.35562622676/4.40031821 ) / ( 0.00664142923*4.63816120529+0.00664142923*7.35562622676+0.01932520196*7.35562622676 )  
        ], #8.00061843202
        [
            9.0 * ( 1.0*0.0158301388*4.63816120529/0.9979354 + 2.0*0.0158301388*7.35562622676/1.58261851 + 4.0*0.04022670088*7.35562622676/4.40040611 ) / ( 0.0158301388*4.63816120529+0.0158301388*7.35562622676+0.04022670088*7.35562622676 )  
        ]  #9.07293374386
    ]   
    
    for k,l in itertools.product(range(0,3),range(0,1)):
        assert abs(S_updated[k][l] - nmtf.S[k][l]) < 0.0000001
        
        
    ###### Test updating S first, then F and G
    R = numpy.array([[1,2],[3,4]],dtype='f')
    M = numpy.array([[1,1],[0,1]])
    K = 3
    L = 1
    
    F = numpy.array([[1,2,3],[4,5,6]],dtype=float)
    S = numpy.array([[7],[8],[9]],dtype=float)
    G = numpy.array([[10],[11]],dtype=float)
    #FSG = numpy.array([[500,550],[1220,1342]],dtype=float)
    #FS = numpy.array([[50],[122]],dtype=float)
    #SG = numpy.array([[70,77],[80,88],[90,99]],dtype=float)
    
    nmtf = NMTF(R,M,K,L)
    nmtf.F = numpy.copy(F)
    nmtf.S = numpy.copy(S)
    nmtf.G = numpy.copy(G) 
    
    nmtf.run(1,0,0,True)
    
    # Updates to S
    # Before updates, R_pred = [[500,550],[1220,1342]]
    # After updating S_00, R_pred = [[430.2058512,473.22643632],[940.82340479,1034.90574527]]
    # After updating S_10, R_pred = [[270.7865487,297.86520357],[542.27514856,596.50266341]]
    S_updated = [
        [
            7.0 * ( 1.0*1.0*10.0/500.0 + 2.0*1.0*11.0/550.0 + 4.0*4.0*11.0/1342.0 ) / ( 1.0*10.0 + 1.0*11.0 + 4.0*11.0 ) 
        ], #0.020585119798234554
        [
            8.0 * ( 1.0*2.0*10.0/430.2058512 + 2.0*2.0*11.0/473.22643632 + 4.0*5.0*11.0/1034.90574527 ) / ( 2.0*10.0 + 2.0*11.0 + 5.0*11.0 )  
        ], #0.029034875264098196
        [
            9.0 * ( 1.0*3.0*10.0/270.7865487 + 2.0*3.0*11.0/297.86520357 + 4.0*6.0*11.0/596.50266341 ) / ( 3.0*10.0 + 3.0*11.0 + 6.0*11.0 )  
        ]  #0.05406592023718962
    ]   
    
    for k,l in itertools.product(range(0,3),range(0,1)):
        assert abs(S_updated[k][l] - nmtf.S[k][l]) < 0.000000000001
    
    # Updates to F
    # Before updates, R_pred = [[2.40852631,2.64937894],[ 5.51910377,6.07101415]])
    # SG = [[0.2058512,0.22643632],[0.29034875,0.31938363],[0.5406592,0.59472512]]
    # After updating F_00, R_pred = [[2.32477182,2.55724901],[5.51910377,6.07101415]]
    # After updating F_01, R_pred = [[2.10091272,2.31100399],[5.51910377,6.07101415]]
    # After updating F_10, R_pred = [[1.58184189,1.74002608],[5.23821446,5.76203591]]
    # After updating F_11, R_pred = [[1.58184189,1.74002608],[4.79426987,5.27369686]]
    F_updated = [[
        1.0 * ( 1.0*0.2058512/2.40852631 + 2.0*0.22643632/2.64937894 ) / ( 0.2058512+0.22643632 ),      #0.5931309211776481
        2.0 * ( 1.0*0.29034875/2.32477182 + 2.0*0.31938363/2.55724901 ) / ( 0.29034875+0.31938363 ),    #1.2289992647854495
        3.0 * ( 1.0*0.5406592/2.10091272 + 2.0*0.59472512/2.31100399 ) / ( 0.5406592+0.59472512 )       #2.0399297159697944
    ],[
        4.0 * 4.0 / 6.07101415,  #2.635474008901791
        5.0 * 4.0 / 5.76203591,  #3.4709953760076444
        6.0 * 4.0 / 5.27369686   #4.5508872878218485
    ]]
    
    for i,k in itertools.product(range(0,2),range(0,3)):
        assert abs(F_updated[i][k] - nmtf.F[i][k]) < 0.00000001
    
    # Updates to G
    # Before updates, R_pred = [[1.58184189,1.74002608],[4.01079375,4.41187313]]
    # FS = [[0.15818419],[0.40107938]]
    # After updating G_00, R_pred = [[1.,1.74002608],[2.53552127,4.41187313]]
    G_updated = [[
        10.0 * 1.0 / 1.58184189   #6.321744330591725  
    ],[
        11.0 * ( 2.0*0.15818419/1.74002608 + 4.0*0.40107938/4.41187313 ) / ( 0.15818419+0.40107938 )  #10.72839422890767
    ]]
    
    for j,l in itertools.product(range(0,2),range(0,1)):
        assert abs(G_updated[j][l] - nmtf.G[j][l]) < 0.0000001
        

    # Also check the stopping criterion of epsilon
    nmtf = NMTF(R,M,K,L)
    nmtf.F = numpy.copy(F)
    nmtf.S = numpy.copy(S)
    nmtf.G = numpy.copy(G) 
    
    nmtf.run(3,0,0.1,True) 
    # After iteration 2, diff_MSE = 0.050005608279011717 < 0.1
    assert nmtf.diff_MSE == 0.050005608279011717
    
    
    # Finally, a quick test whether nmtf.train works as well
    random.seed(0)
    nmtf = NMTF(R,M,K,L)
    nmtf.train(max_iterations=5,updates=1,epsilon_stop=0.05,Kmeans=True,S_random=False,S_first=True)
    # After iteration 2, diff_MSE = 0.032704629007517153 < 0.05
    assert nmtf.diff_MSE == 0.045276282041331516

    
""" Test divergence, MSE, R^2, Rp calculations, as well as computing R_pred """
def test_compute_statistics():
    R = numpy.array([[1,2],[3,4]],dtype=float)
    M = numpy.array([[1,1],[0,1]])
    K = 3
    L = 1
    
    F = numpy.array([[1,2,3],[4,5,6]])
    S = numpy.array([[7],[8],[9]])
    G = numpy.array([[10],[11]])
    R_predicted = numpy.array([[500,550],[1220,1342]],dtype=float)
    
    MSE = ( 499*499 + 548*548 + 1338*1338 ) / 3.0
    
    I_div = sum([
        1.0*math.log(1.0/500.0) - 1.0 + 500.0,
        2.0*math.log(2.0/550.0) - 2.0 + 550.0,
        4.0*math.log(4.0/1342.0) - 4.0 + 1342.0
    ])
    
    R2 = 1 - (499**2 + 548**2 + 1338**2)/((4.0/3.0)**2 + (1.0/3.0)**2 + (5.0/3.0)**2) #mean=7.0/3.0
    #mean_real=7.0/3.0,mean_pred=2392.0/3.0 -> diff_real=[-4.0/3.0,-1.0/3.0,5.0/3.0],diff_pred=[-892.0/3.0,-742.0/3.0,1634.0/3.0]
    Rp = ((-4.0/3.0*-892.0/3.0)+(-1.0/3.0*-742.0/3.0)+(5.0/3.0*1634.0/3.0)) / (math.sqrt((-4.0/3.0)**2+(-1.0/3.0)**2+(5.0/3.0)**2) * math.sqrt((-892.0/3.0)**2+(-742.0/3.0)**2+(1634.0/3.0)**2))
      
    nmtf = NMTF(R,M,K,L)
    nmtf.F = F
    nmtf.S = S
    nmtf.G = G    
    nmtf.M_test = None
    nmtf.compute_statistics()
    
    assert numpy.array_equal(R_predicted,nmtf.R_pred)    
    assert MSE == nmtf.MSE 
    assert I_div == nmtf.I_div   
    assert abs(R2 - nmtf.R2) < 0.000000001
    assert abs(Rp - nmtf.Rp) < 0.000000000000001
    
    assert MSE == nmtf.compute_MSE(M,R,R_predicted)
    assert I_div == nmtf.compute_I_div(M,R,R_predicted) 
    assert abs(R2 - nmtf.compute_R2(M,R,R_predicted)) < 0.000000001
    assert abs(Rp - nmtf.compute_Rp(M,R,R_predicted)) < 0.000000000000001
    assert nmtf.diff_MSE == -779849.66666666663
    
    assert numpy.array_equal(nmtf.all_diff_MSE,[-779849.66666666663])
    assert numpy.array_equal(nmtf.all_MSE,[779849.66666666663])
    assert numpy.array_equal(nmtf.all_R2,[-501330.92857142864])
    
    # Now see if it computes the performance on M_test as well
    M_test = numpy.array([[0,0],[1,1]])
    MSE_pred = (1217**2 + 1338**2) / 2.0
    R2_pred = 1. - (1217**2+1338**2)/(0.5**2+0.5**2) #mean=3.5
    
    nmtf = NMTF(R,M,K,L)
    nmtf.F = F
    nmtf.S = S
    nmtf.G = G
    nmtf.M_test = M_test
    nmtf.compute_statistics()
    assert MSE_pred == nmtf.MSE_pred
    assert R2_pred == nmtf.R2_pred
    
    assert numpy.array_equal(nmtf.all_MSE_pred,[MSE_pred])
    assert numpy.array_equal(nmtf.all_R2_pred,[R2_pred])
    