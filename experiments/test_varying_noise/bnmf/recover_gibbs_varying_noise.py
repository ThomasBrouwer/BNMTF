"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the fraction of entries that are missing, and the amount of noise.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K = 100, 50, 10

The noise levels indicate the percentage of noise, compared to the amount of 
variance in the dataset - i.e. the inverse of the Signal to Noise ratio:
    SNR = std_signal^2 / std_noise^2
    noise = 1 / SNR
We test it for 1%, 10%, 20%, 50%, and 100% noise.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmf_gibbs import bnmf_gibbs
from BNMTF.example.generate_toy.bnmf.generate_bnmf import generate_dataset, add_noise, try_generate_M
from ml_helpers.code.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9 ]
noise_ratios = [ 0.01, 0.05, 0.1 ] # 1/SNR

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 2000
I,J,K = 50, 50, 10
attempts = 1000 # How many attempts we should make at generating M's

burn_in = 1000
thinning = 10

alpha, beta = 1., 1.
lambdaU = numpy.ones((I,K))
lambdaV = numpy.ones((J,K))  
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

'''
# For each tau, generate a dataset, and mask matrices
all_R = []
all_Ms = []
all_Ms_test = []
for noise in noise_ratios:
    (_,_,_,true_R,_) = generate_dataset(I,J,K,lambdaU,lambdaV,alpha/beta)
    
    variance_signal = true_R.var()
    tau = 1. / (variance_signal * noise)
    print "Noise: %s%%. Variance in dataset is %s. Adding noise with variance %s." % (100.*noise,variance_signal,1./tau)
    
    R = add_noise(true_R,tau)
    
    Ms = [ try_generate_M(I,J,fraction,attempts) for fraction in fractions_unknown ]
    Ms_test = [ calc_inverse_M(M) for M in Ms ]
    
    all_R.append(R)
    all_Ms.append(Ms)
    all_Ms_test.append(Ms_test)
    
    # Make sure each M has no empty rows or columns
    def check_empty_rows_columns(M,fraction):
        sums_columns = M.sum(axis=0)
        sums_rows = M.sum(axis=1)
        for i,c in enumerate(sums_rows):
            assert c != 0, "Fully unobserved row in M, row %s. Fraction %s." % (i,fraction)
        for j,c in enumerate(sums_columns):
            assert c != 0, "Fully unobserved column in M, column %s. Fraction %s." % (j,fraction)
            
    for M,fraction in zip(Ms,fractions_unknown):
        check_empty_rows_columns(M,fraction)
        
print "Generated datasets and mask matrices."
       
       
# For each level of noise, and for each fraction of missing data, run the VB algorithm.
# Then plot this, and convergence of tau as well.
all_performances_per_noise = []
for noise,R in zip(noise_ratios,all_R):
    all_performances = []    
    all_taus = []
    
    for fraction,M,M_test in zip(fractions_unknown,Ms,Ms_test):
        print "Trying fraction %s with noise level %s." % (fraction,noise)
        
        # Run the Gibbs sampler
        BNMF = bnmf_gibbs(R,M,K,priors)
        BNMF.initialise()
        BNMF.run(iterations)
        
        all_taus.append(BNMF.all_tau)
        
        # Measure the performances
        performances = BNMF.predict(M_test,burn_in,thinning)
        all_performances.append(performances)
        
    all_performances_per_noise.append(all_performances)
    
    # Plot the tau values to check convergence
    f2, axarr2 = plt.subplots(len(fractions_unknown), sharex=True)
    x2 = range(1,len(all_taus[0])+1)
    axarr2[0].set_title('Convergence of expectations tau with noise level %s' % noise)
    for i,taus in enumerate(all_taus):
        axarr2[i].plot(x2, taus)
        axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
    axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")
    

print "All performances versus noise level and fraction of entries missing: %s." \
    % zip(noise_ratios,[zip(fractions_unknown,all_performances) for all_performances in all_performances_per_noise])


'''
# All performances versus noise level and fraction of entries missing: 
# (Noise -> Fraction -> Performances)
stored_all_performances = [
 (0.01, 
      [(0.1, {'R^2': 0.985903298266529, 'MSE': 0.52335367774259334, 'Rp': 0.99304725608282396}), 
       (0.2, {'R^2': 0.980763727512887, 'MSE': 0.63445170601562029, 'Rp': 0.99056887342912914}), 
       (0.3, {'R^2': 0.9759260743403043, 'MSE': 0.73671405908868126, 'Rp': 0.98805429862029526}), 
       (0.4, {'R^2': 0.9763462565212224, 'MSE': 0.80088628773256965, 'Rp': 0.98812060920882061}), 
       (0.5, {'R^2': 0.9603099919325674, 'MSE': 1.2819942329124554, 'Rp': 0.98022237320919381}), 
       (0.6, {'R^2': 0.9340554816558071, 'MSE': 2.0620582918129817, 'Rp': 0.96656625508931204}), 
       (0.65, {'R^2': 0.8755523825326187, 'MSE': 4.0171576509734921, 'Rp': 0.93586467689519626}), 
       (0.7, {'R^2': 0.8144001289912502, 'MSE': 5.828664073581824, 'Rp': 0.9024481465715557}), 
       (0.75, {'R^2': 0.7409626017625912, 'MSE': 8.3144960084449799, 'Rp': 0.86103399330218722}), 
       (0.8, {'R^2': 0.7106426687917611, 'MSE': 9.0946188859656658, 'Rp': 0.84380445214454181}), 
       (0.85, {'R^2': 0.6466346415062064, 'MSE': 10.881648565859551, 'Rp': 0.80448092691982864}), 
       (0.9, {'R^2': 0.6538566363240571, 'MSE': 10.557955551929794, 'Rp': 0.80880196390796799})]), 
 (0.05, 
      [(0.1, {'R^2': 0.916621165871173, 'MSE': 2.9550337817316796, 'Rp': 0.95789589311072909}), 
       (0.2, {'R^2': 0.8966527629847163, 'MSE': 3.5664041832415112, 'Rp': 0.94712047238042429}), 
       (0.3, {'R^2': 0.9046154167407594, 'MSE': 3.3387439785567383, 'Rp': 0.95119861599753308}), 
       (0.4, {'R^2': 0.8722524697611271, 'MSE': 4.9357471890632221, 'Rp': 0.9339859512310078}), 
       (0.5, {'R^2': 0.8433948514954184, 'MSE': 5.4400497149760749, 'Rp': 0.91936099053545095}), 
       (0.6, {'R^2': 0.6934919870386858, 'MSE': 10.905196718381163, 'Rp': 0.83472342796108512}), 
       (0.65, {'R^2': 0.6472297245950314, 'MSE': 13.207421358934283, 'Rp': 0.80708054423391007}), 
       (0.7, {'R^2': 0.584048422669815, 'MSE': 15.049169083930016, 'Rp': 0.76938127594166439}), 
       (0.75, {'R^2': 0.5972948290020943, 'MSE': 14.648764894400312, 'Rp': 0.77322587194743153}), 
       (0.8, {'R^2': 0.5658318897502532, 'MSE': 15.850466728842202, 'Rp': 0.75356678912700337}), 
       (0.85, {'R^2': 0.4882068126257707, 'MSE': 18.642508641349227, 'Rp': 0.70042192519595747}), 
       (0.9, {'R^2': 0.46949507520735523, 'MSE': 19.583131668592856, 'Rp': 0.69333647403101362})]), 
 (0.1, 
      [(0.1, {'R^2': 0.8601242332662667, 'MSE': 6.4658145640451128, 'Rp': 0.9288604272037766}), 
       (0.2, {'R^2': 0.8133538845149939, 'MSE': 6.6391681783379797, 'Rp': 0.90236574853170115}), 
       (0.3, {'R^2': 0.8045190035698141, 'MSE': 8.5494206671367579, 'Rp': 0.89728999333357429}), 
       (0.4, {'R^2': 0.7551859408255522, 'MSE': 9.7007259688849388, 'Rp': 0.86926570450091789}), 
       (0.5, {'R^2': 0.7384113732970214, 'MSE': 10.439380448627963, 'Rp': 0.85936607601644666}), 
       (0.6, {'R^2': 0.6648032963440703, 'MSE': 14.032485943920898, 'Rp': 0.81548086857225754}), 
       (0.65, {'R^2': 0.6046844363759465, 'MSE': 15.087888841322266, 'Rp': 0.77790035011493841}), 
       (0.7, {'R^2': 0.6040490901396769, 'MSE': 16.029471854061896, 'Rp': 0.77963590248705217}), 
       (0.75, {'R^2': 0.5579885175970947, 'MSE': 18.232462137632488, 'Rp': 0.74739054756936685}), 
       (0.8, {'R^2': 0.5234983900741956, 'MSE': 18.520051005615986, 'Rp': 0.72536539246863374}), 
       (0.85, {'R^2': 0.4672555363111337, 'MSE': 21.404126437164386, 'Rp': 0.68926418420747826}), 
       (0.9, {'R^2': 0.4075669800307794, 'MSE': 24.896757849814581, 'Rp': 0.64275033317289132})])
]
all_performances_per_noise = [[performance for fraction,performance in all_performances] for noise,all_performances in stored_all_performances]
''''''


# Plot the MSE and R^2 for each noise level, as the fraction of missing values varies.
# So we get n lines in each plot for n noise levels.
#f, axarr = plt.subplots(3, sharex=True)
f, axarr = plt.subplots(2, sharex=True, figsize=(7.5,5))
x = fractions_unknown

# Set axis font to small
fontP = FontProperties()
fontP.set_size('small')

axarr[0].set_title('BNMF performance versus fraction missing')
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[0].plot(x, [perf['MSE'] for perf in all_performances],label="Noise %s%%" % (100.*noise))
axarr[0].legend(loc="upper left", prop = fontP)
axarr[0].set_ylabel("MSE")

for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[1].plot(x, [perf['R^2'] for perf in all_performances],label="Noise %s%%" % (100.*noise))   
axarr[1].set_ylabel("R^2")
axarr[1].set_ylim(0.4,1)

#for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
#    axarr[2].plot(x, [perf['Rp'] for perf in all_performances],label="Noise %s%%" % (100.*noise)) 
#axarr[2].set_ylabel("Rp")

#axarr[2].set_xlabel("Fraction missing")
axarr[1].set_xlabel("Fraction missing")
plt.xlim(x[0],x[-1])