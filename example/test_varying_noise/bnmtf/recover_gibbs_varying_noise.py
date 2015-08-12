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
from BNMTF.example.generate_toy.bnmtf.generate_bnmtf import generate_dataset, add_noise
from ml_helpers.code.mask import generate_M, calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fractions_unknown = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8 ]
noise_ratios = [ 0.01, 0.1, 0.2, 0.5, 1. ] # 1/SNR

input_folder = project_location+"BNMTF/example/generate_toy/bnmtf/"

iterations = 2000
I,J,K,L = 100, 50, 10, 5

burn_in = 1000
thinning = 10

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
    [(0.01, 
          [(0.1, {'R^2': 0.9871247168758106, 'MSE': 11.40060696582403, 'Rp': 0.99367434520544251}), 
           (0.2, {'R^2': 0.9868763676241317, 'MSE': 11.985861363068047, 'Rp': 0.99342239221977846}), 
           (0.3, {'R^2': 0.9863900788183456, 'MSE': 11.886405824149346, 'Rp': 0.99318715191538121}), 
           (0.4, {'R^2': 0.9842357594246902, 'MSE': 15.256484492882834, 'Rp': 0.99222380662114251}), 
           (0.5, {'R^2': 0.9844586792569441, 'MSE': 14.036481205772125, 'Rp': 0.99224227344315097}), 
           (0.6, {'R^2': 0.9832429529483072, 'MSE': 15.976341846439908, 'Rp': 0.99159520751320873}), 
           (0.65, {'R^2': 0.9806186050962236, 'MSE': 18.061632044043414, 'Rp': 0.9903755828095433}), 
           (0.7, {'R^2': 0.9794454470643703, 'MSE': 19.404098372659373, 'Rp': 0.9897536753994981}), 
           (0.75, {'R^2': 0.9733140344789216, 'MSE': 25.36436042951366, 'Rp': 0.98659413617376746}), 
           (0.8, {'R^2': 0.9634146031031825, 'MSE': 33.508586980082029, 'Rp': 0.98177541897658049})]), 
     (0.1, 
          [(0.1, {'R^2': 0.902727089907587, 'MSE': 98.883552889510298, 'Rp': 0.9507685250567508}), 
           (0.2, {'R^2': 0.8807046829527929, 'MSE': 93.607581979451865, 'Rp': 0.93855449340313646}), 
           (0.3, {'R^2': 0.8828413274988978, 'MSE': 96.118831499069387, 'Rp': 0.94013788031919243}), 
           (0.4, {'R^2': 0.8788394875599201, 'MSE': 98.418044982897413, 'Rp': 0.93873590335101664}), 
           (0.5, {'R^2': 0.8760506991881979, 'MSE': 103.86191009920026, 'Rp': 0.93674750643454829}), 
           (0.6, {'R^2': 0.858557803887021, 'MSE': 119.27276520448285, 'Rp': 0.9272902984711574}), 
           (0.65, {'R^2': 0.868853725085067, 'MSE': 111.99982261152925, 'Rp': 0.9330008108945439}), 
           (0.7, {'R^2': 0.8648224473843618, 'MSE': 117.37122376109416, 'Rp': 0.93044650023467745}), 
           (0.75, {'R^2': 0.8501965407413636, 'MSE': 128.43346828645585, 'Rp': 0.92302064085150359}), 
           (0.8, {'R^2': 0.8433444557209573, 'MSE': 133.47798772569794, 'Rp': 0.92044872432617986})]), 
     (0.2, 
          [(0.1, {'R^2': 0.8086707408090495, 'MSE': 117.49254988828559, 'Rp': 0.89947745066279083}), 
           (0.2, {'R^2': 0.823290928141119, 'MSE': 125.24723014003855, 'Rp': 0.90910492833043466}), 
           (0.3, {'R^2': 0.8043561602731344, 'MSE': 127.76241449583709, 'Rp': 0.89732306456425959}), 
           (0.4, {'R^2': 0.8085642430728078, 'MSE': 123.05902736468197, 'Rp': 0.89987944340988579}), 
           (0.5, {'R^2': 0.8014522530350684, 'MSE': 129.91794011236522, 'Rp': 0.89707201443983764}), 
           (0.6, {'R^2': 0.786907467055314, 'MSE': 136.94099741754749, 'Rp': 0.88826180814941069}), 
           (0.65, {'R^2': 0.773741027230773, 'MSE': 146.7989227438361, 'Rp': 0.88086158545773263}), 
           (0.7, {'R^2': 0.7726930552671156, 'MSE': 146.69579075177222, 'Rp': 0.8814127467261631}), 
           (0.75, {'R^2': 0.775580060527582, 'MSE': 142.98482297040636, 'Rp': 0.88080763416506502}), 
           (0.8, {'R^2': 0.7694200513064895, 'MSE': 149.7589232382638, 'Rp': 0.8786358776935953})]), 
     (0.5, 
          [(0.1, {'R^2': 0.6529090455181081, 'MSE': 466.69555753851682, 'Rp': 0.81079623155000369}), 
           (0.2, {'R^2': 0.6384974095490453, 'MSE': 491.99984316903709, 'Rp': 0.79946522456459646}), 
           (0.3, {'R^2': 0.6206394224792425, 'MSE': 507.07874827690353, 'Rp': 0.78991051332077966}), 
           (0.4, {'R^2': 0.612764970877818, 'MSE': 512.09784766495829, 'Rp': 0.7840429037440827}), 
           (0.5, {'R^2': 0.6222111922988184, 'MSE': 518.72613290243123, 'Rp': 0.79103786814007082}), 
           (0.6, {'R^2': 0.6225905548358887, 'MSE': 522.90782888498666, 'Rp': 0.79058296815033002}), 
           (0.65, {'R^2': 0.6230103937725399, 'MSE': 530.75388645189696, 'Rp': 0.79154588808085236}), 
           (0.7, {'R^2': 0.6122936051170905, 'MSE': 542.12215478644737, 'Rp': 0.78387012630779063}), 
           (0.75, {'R^2': 0.6005463682690421, 'MSE': 555.02751946377543, 'Rp': 0.77702046349937781}), 
           (0.8, {'R^2': 0.5961193153678095, 'MSE': 554.46792806276028, 'Rp': 0.77534101384933318})]), 
     (1.0, 
          [(0.1, {'R^2': 0.4755469232898373, 'MSE': 767.86621143831269, 'Rp': 0.6931997114675712}), 
           (0.2, {'R^2': 0.46155079621823714, 'MSE': 824.18454722809577, 'Rp': 0.68172457696490496}), 
           (0.3, {'R^2': 0.4522882171357262, 'MSE': 818.28726139335583, 'Rp': 0.67291548363196663}), 
           (0.4, {'R^2': 0.4747669227489426, 'MSE': 848.87296720862412, 'Rp': 0.69477215344888799}), 
           (0.5, {'R^2': 0.4299681443339569, 'MSE': 860.42316292206453, 'Rp': 0.66056716652866965}), 
           (0.6, {'R^2': 0.42733516663392945, 'MSE': 885.37951422897947, 'Rp': 0.65733665466477109}), 
           (0.65, {'R^2': 0.43771640104898113, 'MSE': 870.483118495074, 'Rp': 0.67127576664817201}), 
           (0.7, {'R^2': 0.41280074855833326, 'MSE': 880.07342029167296, 'Rp': 0.64338423683120716}), 
           (0.75, {'R^2': 0.4033045072206155, 'MSE': 893.7894480379307, 'Rp': 0.63679380685137754}), 
           (0.8, {'R^2': 0.40917868101968646, 'MSE': 879.4621923874555, 'Rp': 0.64504622057657535})])]
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
