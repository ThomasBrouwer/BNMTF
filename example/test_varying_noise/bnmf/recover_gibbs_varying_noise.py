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
from BNMTF.example.generate_toy.bnmf.generate_bnmf import generate_dataset, add_noise
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8 ]
noise_ratios = [ 0.01, 0.1, 0.2, 0.5, 1. ] # 1/SNR

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 2000
I,J,K = 100, 50, 10

burn_in = 1000
thinning = 10

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
All performances versus noise level and fraction of entries missing: 
    [(0.01, 
          [(0.1, {'R^2': 0.984327379615597, 'MSE': 0.49291516806919577, 'Rp': 0.9921579363835803}), 
           (0.2, {'R^2': 0.9870850309180961, 'MSE': 0.44439556610330549, 'Rp': 0.99354340355412685}), 
           (0.3, {'R^2': 0.9812581072453863, 'MSE': 0.53557885768291214, 'Rp': 0.99063272657163615}), 
           (0.4, {'R^2': 0.9807507902828302, 'MSE': 0.63251815764731167, 'Rp': 0.99039451606225415}), 
           (0.5, {'R^2': 0.9726986338367348, 'MSE': 0.83166753021130424, 'Rp': 0.98630606561533318}), 
           (0.6, {'R^2': 0.96622277767559, 'MSE': 1.0057840933812459, 'Rp': 0.98298875849883849}), 
           (0.65, {'R^2': 0.9447030408369691, 'MSE': 1.6714684032666085, 'Rp': 0.97212124673649503}), 
           (0.7, {'R^2': 0.9246556824505652, 'MSE': 2.3178317466888712, 'Rp': 0.96168095823041777}), 
           (0.75, {'R^2': 0.8362217267048724, 'MSE': 4.956909843140286, 'Rp': 0.91487143753244848}), 
           (0.8, {'R^2': 0.6663077678209981, 'MSE': 10.333983877157896, 'Rp': 0.81745804540144329})]), 
     (0.1, 
          [(0.1, {'R^2': 0.8765521195731734, 'MSE': 3.5957109788382615, 'Rp': 0.93636719291577586}), 
           (0.2, {'R^2': 0.8697672003885271, 'MSE': 3.6912873902093546, 'Rp': 0.93271016667534179}), 
           (0.3, {'R^2': 0.8515818088520015, 'MSE': 4.3739725603661075, 'Rp': 0.92289109899948152}), 
           (0.4, {'R^2': 0.8247299627282597, 'MSE': 4.8949585759665526, 'Rp': 0.90820309812197098}), 
           (0.5, {'R^2': 0.7974135479757228, 'MSE': 6.002248588016287, 'Rp': 0.89383901831327184}), 
           (0.6, {'R^2': 0.7806310192568402, 'MSE': 6.4464144487347559, 'Rp': 0.88360865011569378}), 
           (0.65, {'R^2': 0.7027422121417745, 'MSE': 8.7631433474405398, 'Rp': 0.83837203619588718}), 
           (0.7, {'R^2': 0.6277023690115331, 'MSE': 10.973282498922208, 'Rp': 0.79388502385910298}), 
           (0.75, {'R^2': 0.6425477034062872, 'MSE': 9.9348965439281702, 'Rp': 0.80177186125298849}), 
           (0.8, {'R^2': 0.5246944505846692, 'MSE': 13.395646663661475, 'Rp': 0.72901884806959749})]), 
     (0.2, 
          [(0.1, {'R^2': 0.7604951052932718, 'MSE': 12.814810990950185, 'Rp': 0.87213662586593366}), 
           (0.2, {'R^2': 0.7272932989242422, 'MSE': 13.164973582134422, 'Rp': 0.85366128733097424}), 
           (0.3, {'R^2': 0.7305258924818199, 'MSE': 13.117962552712095, 'Rp': 0.85498679824798907}), 
           (0.4, {'R^2': 0.6976874780589213, 'MSE': 15.37729133197117, 'Rp': 0.83561833955740383}), 
           (0.5, {'R^2': 0.6842164205122525, 'MSE': 15.339726410904008, 'Rp': 0.82738404741704019}), 
           (0.6, {'R^2': 0.6596859460330464, 'MSE': 18.269520469358255, 'Rp': 0.81583059580429829}), 
           (0.65, {'R^2': 0.6221483565869295, 'MSE': 19.432421157870998, 'Rp': 0.78917034173874612}), 
           (0.7, {'R^2': 0.5968885961352156, 'MSE': 21.075612977663429, 'Rp': 0.77298223629623974}), 
           (0.75, {'R^2': 0.6011369706614151, 'MSE': 20.860140256315873, 'Rp': 0.77570369421660146}), 
           (0.8, {'R^2': 0.5384760323596918, 'MSE': 24.792870144462448, 'Rp': 0.73422423345262522})]), 
     (0.5, 
          [(0.1, {'R^2': 0.5080866452807974, 'MSE': 19.854864835355144, 'Rp': 0.71408421636320296}), 
           (0.2, {'R^2': 0.5469940113575574, 'MSE': 21.570416897359063, 'Rp': 0.73970783440789267}), 
           (0.3, {'R^2': 0.5280346547679874, 'MSE': 23.508103570280092, 'Rp': 0.72675835551051116}), 
           (0.4, {'R^2': 0.4982668489455703, 'MSE': 22.915359484489098, 'Rp': 0.70653872312705579}), 
           (0.5, {'R^2': 0.4456510022242025, 'MSE': 24.426559048006446, 'Rp': 0.66899260762554791}), 
           (0.6, {'R^2': 0.4494714649967708, 'MSE': 25.21973716162924, 'Rp': 0.67138894486907608}), 
           (0.65, {'R^2': 0.45243904882791297, 'MSE': 26.56083812327492, 'Rp': 0.67395658733263597}), 
           (0.7, {'R^2': 0.423811224513171, 'MSE': 26.651836173601627, 'Rp': 0.65356769986976948}), 
           (0.75, {'R^2': 0.4308135511516591, 'MSE': 26.590121792683547, 'Rp': 0.65687793323933164}), 
           (0.8, {'R^2': 0.41405488636188603, 'MSE': 27.730466368455247, 'Rp': 0.64424334486074208})]), 
     (1.0, 
          [(0.1, {'R^2': 0.39213998134514516, 'MSE': 35.189904426450717, 'Rp': 0.62965470060440776}), 
           (0.2, {'R^2': 0.40255383127054656, 'MSE': 32.856073599394321, 'Rp': 0.63475216257445477}), 
           (0.3, {'R^2': 0.3326498487652867, 'MSE': 34.097063788125361, 'Rp': 0.57717080093849571}), 
           (0.4, {'R^2': 0.3148844307551283, 'MSE': 36.377367119950065, 'Rp': 0.56116503537502838}), 
           (0.5, {'R^2': 0.32426420141027057, 'MSE': 33.796819097670074, 'Rp': 0.56952099680697854}), 
           (0.6, {'R^2': 0.29632056298946297, 'MSE': 36.388812071192959, 'Rp': 0.54442281767556966}), 
           (0.65, {'R^2': 0.2923214991044293, 'MSE': 37.363776253296038, 'Rp': 0.54100644901875194}), 
           (0.7, {'R^2': 0.27204738354340796, 'MSE': 37.810432045864637, 'Rp': 0.52160548989200872}), 
           (0.75, {'R^2': 0.26117287440602965, 'MSE': 37.375712885151991, 'Rp': 0.51210475445738457}), 
           (0.8, {'R^2': 0.24695090768722028, 'MSE': 39.401986915829767, 'Rp': 0.49919191756447562})])]
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
