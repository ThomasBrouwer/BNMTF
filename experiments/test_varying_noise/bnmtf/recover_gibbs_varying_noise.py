"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the fraction of entries that are missing.
We repeat this 10 times per fraction and average that.

We use the correct number of latent factors and flatter priors than used to generate the data.

I, J, K, L = 100, 80, 5, 5

The noise levels indicate the percentage of noise, compared to the amount of 
variance in the dataset - i.e. the inverse of the Signal to Noise ratio:
    SNR = std_signal^2 / std_noise^2
    noise = 1 / SNR
We test it for 1%, 10%, 20%, 50%, and 100% noise.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.experiments.generate_toy.bnmtf.generate_bnmtf import add_noise, try_generate_M
from ml_helpers.code.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fraction_unknown = 0.1
noise_ratios = [ 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1. ] # 1/SNR

input_folder = project_location+"BNMTF/experiments/generate_toy/bnmtf/"

repeats = 10
iterations = 1000
burn_in = 800
thinning = 5

init_S = 'random'
init_FG = 'kmeans'
I,J,K,L = 100, 80, 5, 5

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))/10.
lambdaS = numpy.ones((K,L))/10.
lambdaG = numpy.ones((J,L))/10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

metrics = ['MSE', 'R^2', 'Rp']


# Load in data
R_true = numpy.loadtxt(input_folder+"R_true.txt")


# For each noise ratio, generate mask matrices for each attempt
M_attempts = 100
all_Ms = [ 
    [try_generate_M(I,J,fraction_unknown,M_attempts) for r in range(0,repeats)]
    for noise in noise_ratios
]
all_Ms_test = [ [calc_inverse_M(M) for M in Ms] for Ms in all_Ms ]

# Make sure each M has no empty rows or columns
def check_empty_rows_columns(M,fraction):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
    for i,c in enumerate(sums_rows):
        assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
    for j,c in enumerate(sums_columns):
        assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
        
for Ms in all_Ms:
    for M in Ms:
        check_empty_rows_columns(M,fraction_unknown)


# For each noise ratio, add that level of noise to the true R
all_R = []
variance_signal = R_true.var()
for noise in noise_ratios:
    tau = 1. / (variance_signal * noise)
    print "Noise: %s%%. Variance in dataset is %s. Adding noise with variance %s." % (100.*noise,variance_signal,1./tau)
    
    R = add_noise(R_true,tau)
    all_R.append(R)
    
    
# We now run the VB algorithm on each of the M's for each noise ratio    
all_performances = {metric:[] for metric in metrics} 
average_performances = {metric:[] for metric in metrics} # averaged over repeats
for (noise,R,Ms,Ms_test) in zip(noise_ratios,all_R,all_Ms,all_Ms_test):
    print "Trying noise ratio %s." % noise
    
    # Run the algorithm <repeats> times and store all the performances
    for metric in metrics:
        all_performances[metric].append([])
    for (repeat,M,M_test) in zip(range(0,repeats),Ms,Ms_test):
        print "Repeat %s of noise ratio %s." % (repeat+1, noise)
    
        BNMF = bnmtf_gibbs_optimised(R,M,K,L,priors)
        BNMF.initialise(init_S,init_FG)
        BNMF.run(iterations)
    
        # Measure the performances
        performances = BNMF.predict(M_test,burn_in,thinning)
        for metric in metrics:
            # Add this metric's performance to the list of <repeat> performances for this noise ratio
            all_performances[metric][-1].append(performances[metric])
            
    # Compute the average across attempts
    for metric in metrics:
        average_performances[metric].append(sum(all_performances[metric][-1])/repeats)
    

    
print "repeats=%s \nnoise_ratios = %s \nall_performances = %s \naverage_performances = %s" % \
    (repeats,noise_ratios,all_performances,average_performances)


'''
repeats=10 
noise_ratios = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0] 
all_performances = {'R^2': [[0.9999762051562596, 0.9999803399582806, 0.9999777589782466, 0.9999709683708068, 0.999978340629859, 0.9999852791454913, 0.9999780394376246, 0.9999780429119534, 0.9999809961629468, 0.9999761694513424], [0.9892731322950635, 0.9882854518233279, 0.9900063397701583, 0.9881239121668208, 0.9869770339990847, 0.9868975318598722, 0.9866051781655802, 0.9883115546912089, 0.9861865655688425, 0.9874666234275644], [0.9438952141317052, 0.9490986446107796, 0.9501474627078781, 0.9491475620212295, 0.946260085022051, 0.9445842470668413, 0.9428754789613143, 0.9441871319960652, 0.9561083936916753, 0.9399280922411841], [0.9034148772718399, 0.8610744113070802, 0.8879519872745595, 0.916425067644823, 0.9012842171539925, 0.9085575829562682, 0.9244983175454888, 0.8817365399073894, 0.9027911239718345, 0.9129670130408155], [0.7952883145845983, 0.8427627550119179, 0.8429012794541995, 0.7670370800835049, 0.7966206655974436, 0.8306180239099952, 0.8340723236334684, 0.793407942386595, 0.7929029092002904, 0.7633524795396956], [0.597888005052394, 0.6523041696908087, 0.6246973351444023, 0.6301927676172854, 0.637402901709756, 0.6957618471471142, 0.5989932585319715, 0.6286019864387988, 0.6084617880901084, 0.6442493328175058], [0.5048388686215449, 0.47643185345917827, 0.4790398469198044, 0.4505495904416955, 0.4314043896214116, 0.3931817106175245, 0.44481216051759753, 0.47914691175030166, 0.46295347767631867, 0.474770295656618]], 'MSE': [[0.012415082985115884, 0.012306805625041068, 0.012594610111962844, 0.013270684282550224, 0.010150643671264699, 0.0089559724419040546, 0.011728490766570924, 0.0090774206002179562, 0.010051131432807723, 0.014910134987471551], [6.3650250745827917, 6.9772875134660861, 6.4605202277868203, 6.1019213235183258, 7.0274634574243411, 7.0954735436180609, 6.5175028912503032, 6.2272219239073845, 6.8898068310467746, 6.3638177906839566], [32.010298177202706, 31.487229736120653, 31.810889451524059, 29.4242904020069, 34.728693798067908, 31.382982888168932, 34.257787907612325, 32.870050586180923, 30.819384758479522, 32.034305900309292], [63.323310633723601, 70.602058737926995, 69.871820360918065, 58.881884141925894, 64.522029966843533, 64.295549232528842, 64.351886498402834, 67.461727785972201, 66.40112989798331, 60.956783887618329], [126.81416740422681, 124.84229909106757, 123.80205590433334, 129.82465610596631, 132.71500841111327, 110.11622066240963, 116.89359328338013, 123.38412528357388, 133.61574981677359, 140.56154375898188], [308.0042624208412, 303.14321326828451, 318.19342650156244, 344.97547455267596, 320.97018624061127, 307.78326016347887, 327.29382926466985, 312.31221570395195, 348.79997809616282, 303.38845949964548], [639.80911648498068, 613.5268157024218, 635.18753176310156, 597.53728342734826, 691.7763863344012, 647.7311836214451, 599.21570883873733, 644.31787764090848, 609.82917212056236, 645.68868289391617]], 'Rp': [[0.9999881065064895, 0.99999018976224585, 0.99998889239916855, 0.99998568201202054, 0.99998928827804212, 0.99999267942269332, 0.99998903099637304, 0.99998903434209263, 0.99999054909869278, 0.9999880975944504], [0.99464635578861182, 0.99415342613242608, 0.99499239381636762, 0.99405595121104184, 0.99349787233547615, 0.99348160995262647, 0.99329451353996534, 0.9941402292360294, 0.99308142207777561, 0.99373620072851132], [0.97155598274842569, 0.97422744642650227, 0.97485608013488723, 0.97438157263830005, 0.97286250234284333, 0.97193572830412578, 0.97104801932531637, 0.97176713448853169, 0.97783680372231296, 0.96957420980464581], [0.95049426165455497, 0.92830438835740114, 0.94245298491190888, 0.95767448217920093, 0.9494996725421323, 0.95319253944940974, 0.96168209553503103, 0.93905773936876447, 0.95018044859905504, 0.95553520025928307], [0.89213853598626336, 0.91812677938857157, 0.91838975360842035, 0.87653604836900689, 0.89261128945734858, 0.91145394453087225, 0.91377937423229649, 0.89086314002809908, 0.89080416524785622, 0.87384780856985478], [0.77434874913939045, 0.80848453327017944, 0.79128052617029188, 0.79425404964433588, 0.79896751623073836, 0.83422677141147528, 0.77504876533327804, 0.79346132555892146, 0.78069046405720699, 0.8032911117724455], [0.71077968104535472, 0.69118837173023373, 0.69496582313995658, 0.67132582711765865, 0.65975838457075742, 0.63169509349471398, 0.66834413469997389, 0.69282403369600454, 0.68178490055736751, 0.69099890406702391]]} 
average_performances = {'R^2': [0.99997821402028109, 0.98781333237675251, 0.94662323124507231, 0.90007011380740898, 0.80589637734017094, 0.6318553392240146, 0.45971291052819946], 'MSE': [0.011546097690490694, 6.6026040577284846, 32.082591360567321, 65.066818114384361, 126.25694197218263, 319.48643057118841, 632.46197588278233], 'Rp': [0.99998915504122665, 0.99390799748188319, 0.97300454799358926, 0.94880738128567399, 0.89785508394185887, 0.79540538125882621, 0.67936651541190451]}
'''


# Plot the MSE, R^2 and Rp
for metric in metrics:
    plt.figure()
    x = noise_ratios
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("Noise ratios missing")
    plt.ylabel(metric)