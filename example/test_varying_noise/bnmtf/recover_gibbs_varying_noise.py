"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the fraction of entries that are missing, and the amount of noise.

We use the correct number of latent factors and same priors as used to generate the data.

I, J, K, L = 100, 50, 10, 5

The noise levels indicate the percentage of noise, compared to the amount of 
variance in the dataset - i.e. the inverse of the Signal to Noise ratio:
    SNR = std_signal^2 / std_noise^2
    noise = 1 / SNR
We test it for 1%, 10%, 20%, 50%, and 100% noise.
"""

project_location = "/home/tab43/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.bnmtf_gibbs import bnmtf_gibbs
from BNMTF.example.generate_toy.bnmtf.generate_bnmtf import generate_dataset, add_noise, try_generate_M
from ml_helpers.code.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9 ]
noise_ratios = [ 0.01, 0.05, 0.1] # 1/SNR
attempts = 1000 # How many attempts we should make at generating M's

input_folder = project_location+"BNMTF/example/generate_toy/bnmtf/"

iterations = 2000
I,J,K,L = 50, 50, 10, 5

burn_in = 1000
thinning = 5

alpha, beta = 1., 1.
lambdaF = numpy.ones((I,K))
lambdaS = numpy.ones((K,L))  
lambdaG = numpy.ones((J,L))  
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# For each tau, generate a dataset, and mask matrices
all_R = []
all_Ms = []
all_Ms_test = []
for noise in noise_ratios:
    (_,_,_,_,true_R,_) = generate_dataset(I,J,K,L,lambdaF,lambdaS,lambdaG,alpha/beta)
    
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
        BNMF = bnmtf_gibbs(R,M,K,L,priors)
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
All performances versus noise level and fraction of entries missing: 
(Noise -> Fraction -> Performances)
[(0.01, 
      [(0.1, {'R^2': 0.9827806402055835, 'MSE': 19.000913967003534, 'Rp': 0.99135308174986569}), 
       (0.2, {'R^2': 0.9865015312881293, 'MSE': 20.788764048766264, 'Rp': 0.99336726740295134}), 
       (0.3, {'R^2': 0.9863355718902775, 'MSE': 21.898187013256795, 'Rp': 0.99322770486605105}), 
       (0.4, {'R^2': 0.9847792151448014, 'MSE': 21.26651989006281, 'Rp': 0.99240266390293563}), 
       (0.5, {'R^2': 0.9793818460353085, 'MSE': 25.527563713676507, 'Rp': 0.98968088446714408}), 
       (0.6, {'R^2': 0.9771151790439166, 'MSE': 29.491116257093598, 'Rp': 0.98859848203983469}), 
       (0.65, {'R^2': 0.9726174546082151, 'MSE': 32.392013491331518, 'Rp': 0.98742957034439638}), 
       (0.7, {'R^2': 0.9747441958006468, 'MSE': 35.056063772985155, 'Rp': 0.98736335556446253}), 
       (0.75, {'R^2': 0.959424791348921, 'MSE': 61.543903644227001, 'Rp': 0.98027679680869395}), 
       (0.8, {'R^2': 0.9618873146810709, 'MSE': 52.393785855408773, 'Rp': 0.98101442505505265}), 
       (0.85, {'R^2': 0.9414502170410091, 'MSE': 83.857636339557686, 'Rp': 0.97093417146508565}), 
       (0.9, {'R^2': 0.9264034206342515, 'MSE': 107.08548550815796, 'Rp': 0.96361659649615694})]), 
 (0.05, 
      [(0.1, {'R^2': 0.9374647792984951, 'MSE': 55.250483452845394, 'Rp': 0.9688462317906581}), 
       (0.2, {'R^2': 0.9340439192178016, 'MSE': 59.003995538837351, 'Rp': 0.9664935318334863}), 
       (0.3, {'R^2': 0.9338128248315327, 'MSE': 57.345658738782895, 'Rp': 0.96696639926188355}), 
       (0.4, {'R^2': 0.92776395586435, 'MSE': 65.51727082417375, 'Rp': 0.96338403220086233}), 
       (0.5, {'R^2': 0.9131779207717767, 'MSE': 76.070580427924682, 'Rp': 0.95578861034099261}), 
       (0.6, {'R^2': 0.9170476133913802, 'MSE': 77.289037491370678, 'Rp': 0.95914364313270806}), 
       (0.65, {'R^2': 0.9082384741236983, 'MSE': 84.325600914815098, 'Rp': 0.95425897635212065}), 
       (0.7, {'R^2': 0.9019297948448788, 'MSE': 89.07421747317558, 'Rp': 0.95161473982110034}), 
       (0.75, {'R^2': 0.902141181004888, 'MSE': 87.539142856337605, 'Rp': 0.95116366276519049}), 
       (0.8, {'R^2': 0.8912338504827146, 'MSE': 98.091026248012554, 'Rp': 0.94524530528108086}), 
       (0.85, {'R^2': 0.88276661453086, 'MSE': 104.20860624841544, 'Rp': 0.94020172050392081}), 
       (0.9, {'R^2': 0.8551715615973307, 'MSE': 131.5332463620307, 'Rp': 0.9294245707193981})]), 
 (0.1, 
      [(0.1, {'R^2': 0.8781627744909675, 'MSE': 44.040917906527127, 'Rp': 0.94015452963719803}), 
       (0.2, {'R^2': 0.892178875968481, 'MSE': 44.007348517421448, 'Rp': 0.94473939210114366}), 
       (0.3, {'R^2': 0.8731356817558712, 'MSE': 49.154656959609994, 'Rp': 0.93615268299815257}), 
       (0.4, {'R^2': 0.862755551324125, 'MSE': 47.022220776352782, 'Rp': 0.92910095883302091}), 
       (0.5, {'R^2': 0.8656404318821806, 'MSE': 50.278963047349507, 'Rp': 0.93097661082436023}), 
       (0.6, {'R^2': 0.8512810885503155, 'MSE': 54.232335229947985, 'Rp': 0.92428711452759704}), 
       (0.65, {'R^2': 0.8506272457387485, 'MSE': 57.039915361887289, 'Rp': 0.92523871587473971}), 
       (0.7, {'R^2': 0.84984158817395, 'MSE': 55.34831158647016, 'Rp': 0.92386343525173931}), 
       (0.75, {'R^2': 0.8317152863079489, 'MSE': 59.309184298399096, 'Rp': 0.91464996028396983}), 
       (0.8, {'R^2': 0.8348612978982133, 'MSE': 60.229660889051601, 'Rp': 0.91404450206165555}), 
       (0.85, {'R^2': 0.8052573923279411, 'MSE': 69.113408274560896, 'Rp': 0.89901532296961462}), 
       (0.9, {'R^2': 0.7621337376835944, 'MSE': 87.47487375643496, 'Rp': 0.87333697688306988})])]
'''


# Plot the MSE and R^2 for each noise level, as the fraction of missing values varies.
# So we get n lines in each plot for n noise levels.
f, axarr = plt.subplots(3, sharex=True)
x = fractions_unknown

# Set axis font to small
fontP = FontProperties()
fontP.set_size('small')

axarr[0].set_title('BNMTF performance versus fraction missing')
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[0].plot(x, [perf['MSE'] for perf in all_performances],label="Noise %s%%" % (100.*noise))
    axarr[0].legend(loc="upper left", prop = fontP)
axarr[0].set_ylabel("MSE")
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[1].plot(x, [perf['R^2'] for perf in all_performances],label="Noise %s%%" % (100.*noise))   
axarr[1].set_ylabel("R^2")
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[2].plot(x, [perf['Rp'] for perf in all_performances],label="Noise %s%%" % (100.*noise)) 
axarr[2].set_ylabel("Rp")
axarr[2].set_xlabel("Fraction missing")
