"""
Test the performance of Variational Bayes for recovering a toy dataset, where 
we vary the fraction of entries that are missing, and the amount of noise.

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

from BNMTF.code.bnmf_vb import bnmf_vb
from BNMTF.example.generate_toy.bnmf.generate_bnmf import generate_dataset, add_noise
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8 ]
noise_ratios = [ 0.01, 0.1, 0.2, 0.5, 1. ] # 1/SNR

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 1000
I,J,K = 100, 50, 10

alpha, beta = 1., 1.
lambdaU = numpy.ones((I,K))
lambdaV = numpy.ones((J,K))  
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

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
    
    Ms = [ generate_M(I,J,fraction) for fraction in fractions_unknown ]
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
    all_exp_taus = []
    
    for fraction,M,M_test in zip(fractions_unknown,Ms,Ms_test):
        print "Trying fraction %s with noise level %s." % (fraction,noise)
        
        # Run the Gibbs sampler
        BNMF = bnmf_vb(R,M,K,priors)
        BNMF.initialise()
        BNMF.run(iterations)
        
        all_exp_taus.append(BNMF.all_exp_tau)
        
        # Measure the performances
        performances = BNMF.predict(M_test)
        all_performances.append(performances)
        
    all_performances_per_noise.append(all_performances)
    
    # Plot the tau values to check convergence
    f2, axarr2 = plt.subplots(len(fractions_unknown), sharex=True)
    x2 = range(1,len(all_exp_taus[0])+1)
    axarr2[0].set_title('Convergence of expectations tau with noise level %s' % noise)
    for i,taus in enumerate(all_exp_taus):
        axarr2[i].plot(x2, taus)
        axarr2[i].set_ylabel("Fraction %s" % fractions_unknown[i])
    axarr2[len(fractions_unknown)-1].set_xlabel("Iterations")
    

print "All performances versus noise level and fraction of entries missing: %s." \
    % zip(noise_ratios,[zip(fractions_unknown,all_performances) for all_performances in all_performances_per_noise])


'''
All performances versus noise level and fraction of entries missing: 
    [(0.01, 
          [(0.1, {'R^2': 0.9813514113028342, 'MSE': 0.46519571731198306, 'Rp': 0.99067442794248151}), 
           (0.2, {'R^2': 0.9836021377749169, 'MSE': 0.43767609296635923, 'Rp': 0.99183056727404095}), 
           (0.3, {'R^2': 0.9821183627463833, 'MSE': 0.47560766221874073, 'Rp': 0.99106099291142224}), 
           (0.4, {'R^2': 0.9797698468452392, 'MSE': 0.55998143797213018, 'Rp': 0.9898427549665515}), 
           (0.5, {'R^2': 0.9752813347589498, 'MSE': 0.7084988261572196, 'Rp': 0.98763851104532607}), 
           (0.6, {'R^2': 0.9658019875232262, 'MSE': 1.0004155248794668, 'Rp': 0.98279620617542351}), 
           (0.65, {'R^2': 0.8993480809397417, 'MSE': 2.7701583685550983, 'Rp': 0.94899632278357948}), 
           (0.7, {'R^2': 0.8100281505398775, 'MSE': 5.1920019923799616, 'Rp': 0.90268653373999708}), 
           (0.75, {'R^2': 0.7011657736414094, 'MSE': 8.2132317305124207, 'Rp': 0.84125068483195042}), 
           (0.8, {'R^2': 0.6518355062677148, 'MSE': 9.6726726121259237, 'Rp': 0.81207839088435108})]), 
     (0.1, 
          [(0.1, {'R^2': 0.8752862644117714, 'MSE': 3.1944133653345976, 'Rp': 0.9356055948306945}), 
           (0.2, {'R^2': 0.8586842612994123, 'MSE': 3.5733234205236761, 'Rp': 0.92676759244741513}), 
           (0.3, {'R^2': 0.8522091067142935, 'MSE': 3.750698351585998, 'Rp': 0.92337637716904319}), 
           (0.4, {'R^2': 0.8250469556357851, 'MSE': 4.3521220507526399, 'Rp': 0.90861135236116808}), 
           (0.5, {'R^2': 0.8135353131034527, 'MSE': 4.7286670414440142, 'Rp': 0.90255454808274971}), 
           (0.6, {'R^2': 0.7063923486899412, 'MSE': 7.2380461267024803, 'Rp': 0.84314107066483679}), 
           (0.65, {'R^2': 0.6097206931841236, 'MSE': 9.9666361641482144, 'Rp': 0.78636650054530277}), 
           (0.7, {'R^2': 0.5208403884141831, 'MSE': 11.817431542398976, 'Rp': 0.73196242602957096}), 
           (0.75, {'R^2': 0.4644208497179746, 'MSE': 13.628425165727615, 'Rp': 0.68889184319356145}), 
           (0.8, {'R^2': 0.5099958012866396, 'MSE': 12.054322232178622, 'Rp': 0.72297259466029651})]), 
     (0.2, 
          [(0.1, {'R^2': 0.7957358763989716, 'MSE': 6.9008703174757233, 'Rp': 0.89270141958488947}), 
           (0.2, {'R^2': 0.7617660556501171, 'MSE': 7.5384678375366905, 'Rp': 0.8729067235912229}), 
           (0.3, {'R^2': 0.7568066904182987, 'MSE': 7.5845379726301898, 'Rp': 0.87020459518089155}), 
           (0.4, {'R^2': 0.73482238175516, 'MSE': 8.5315638181193556, 'Rp': 0.8574563060559427}), 
           (0.5, {'R^2': 0.6834181664554696, 'MSE': 10.253731191991815, 'Rp': 0.82797035700515609}), 
           (0.6, {'R^2': 0.6207981478357999, 'MSE': 12.524578919084066, 'Rp': 0.79030853615464713}), 
           (0.65, {'R^2': 0.5505919154610631, 'MSE': 14.259652265234134, 'Rp': 0.75272695961261615}), 
           (0.7, {'R^2': 0.5276860640286306, 'MSE': 15.35152524050849, 'Rp': 0.73556320507653805}), 
           (0.75, {'R^2': 0.5056885482807768, 'MSE': 15.405479499080686, 'Rp': 0.71877060189076447}), 
           (0.8, {'R^2': 0.45827686379966137, 'MSE': 17.219390045612222, 'Rp': 0.6933801724470966})]), 
     (0.5, 
          [(0.1, {'R^2': 0.5789789987637582, 'MSE': 18.48431142672824, 'Rp': 0.76564309718879742}), 
           (0.2, {'R^2': 0.5111005357976071, 'MSE': 20.700410600785489, 'Rp': 0.71685518662393211}), 
           (0.3, {'R^2': 0.4836296231676913, 'MSE': 21.248856271226796, 'Rp': 0.69799542300994988}), 
           (0.4, {'R^2': 0.47200296988741186, 'MSE': 24.136448973437673, 'Rp': 0.69132080843198696}), 
           (0.5, {'R^2': 0.4601116683738895, 'MSE': 22.615114726336483, 'Rp': 0.68193251048191994}), 
           (0.6, {'R^2': 0.4391008991885297, 'MSE': 23.534159247514253, 'Rp': 0.66884534168040721}), 
           (0.65, {'R^2': 0.4047218643868299, 'MSE': 25.189960224918494, 'Rp': 0.64201902541640121}), 
           (0.7, {'R^2': 0.3695974988583959, 'MSE': 26.232692669926063, 'Rp': 0.61661478386972535}), 
           (0.75, {'R^2': 0.3798379262380439, 'MSE': 26.064489256250688, 'Rp': 0.62750737425752201}), 
           (0.8, {'R^2': 0.33222069338714577, 'MSE': 28.608484355410067, 'Rp': 0.60651748375210446})]), 
     (1.0, 
          [(0.1, {'R^2': 0.4014396600890896, 'MSE': 21.910642570335153, 'Rp': 0.63446454406263586}), 
           (0.2, {'R^2': 0.318067037779596, 'MSE': 24.124318519753711, 'Rp': 0.57008923420318758}), 
           (0.3, {'R^2': 0.3104620140613499, 'MSE': 25.391758947542431, 'Rp': 0.56895529498156072}), 
           (0.4, {'R^2': 0.26651399339421555, 'MSE': 26.858396043461571, 'Rp': 0.52886994522252173}), 
           (0.5, {'R^2': 0.23555252309532992, 'MSE': 28.430627019637242, 'Rp': 0.4998350011832256}), 
           (0.6, {'R^2': 0.22282087772781234, 'MSE': 28.912465173869254, 'Rp': 0.48313611962313197}), 
           (0.65, {'R^2': 0.20602136983203034, 'MSE': 29.688522877292865, 'Rp': 0.46991404877965076}), 
           (0.7, {'R^2': 0.19685205719184706, 'MSE': 29.213874882969197, 'Rp': 0.46677770729542389}), 
           (0.75, {'R^2': 0.17381120933911443, 'MSE': 30.348405500903272, 'Rp': 0.43737940030792155}), 
           (0.8, {'R^2': 0.17982908825610722, 'MSE': 30.026068094329894, 'Rp': 0.45248178814955542})])].
'''


# Plot the MSE and R^2 for each noise level, as the fraction of missing values varies.
# So we get n lines in each plot for n noise levels.
f, axarr = plt.subplots(3, sharex=True)
x = fractions_unknown
axarr[0].set_title('Performance versus fraction missing')
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[0].plot(x, [perf['MSE'] for perf in all_performances],label="Noise %s%%" % (100.*noise))
    axarr[0].legend(loc="upper left")
axarr[0].set_ylabel("MSE")
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[1].plot(x, [perf['R^2'] for perf in all_performances],label="Noise %s%%" % (100.*noise))   
axarr[1].set_ylabel("R^2")
for noise,all_performances in zip(noise_ratios,all_performances_per_noise):
    axarr[2].plot(x, [perf['Rp'] for perf in all_performances],label="Noise %s%%" % (100.*noise)) 
axarr[2].set_ylabel("Rp")
axarr[2].set_xlabel("Fraction missing")
