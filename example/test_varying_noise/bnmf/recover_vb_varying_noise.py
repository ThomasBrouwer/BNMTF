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

fractions_unknown = [0.8]#[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8 ]
noise_ratios = [ 0.01, 0.1, 0.2, 0.5, 1. ] # 1/SNR

input_folder = project_location+"BNMTF/example/generate_toy/bnmf/"

iterations = 100
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


#For 100 iterations: all_performances_per_noise = [[{'R^2': 0.9847693188327653, 'MSE': 1.6367882723298706, 'Rp': 0.9923617757377502}, {'R^2': 0.9832745054633855, 'MSE': 1.7363129728237157, 'Rp': 0.9916028320368615}, {'R^2': 0.9835033066179951, 'MSE': 2.1789113741766397, 'Rp': 0.99171964140971158}, {'R^2': 0.9771295037506261, 'MSE': 2.5336360816089512, 'Rp': 0.98853541688957325}, {'R^2': 0.9573192689005743, 'MSE': 4.8245922919351925, 'Rp': 0.97874000909663783}, {'R^2': 0.8859401481514391, 'MSE': 13.330122996088159, 'Rp': 0.94196519077610674}, {'R^2': 0.747606174220403, 'MSE': 29.031947411260713, 'Rp': 0.86783309176694234}, {'R^2': 0.5880623812996403, 'MSE': 47.003426993573626, 'Rp': 0.7804121799722844}], [{'R^2': 0.9865716725725598, 'MSE': 1.7994546885878901, 'Rp': 0.99328258741756448}, {'R^2': 0.9843302733221452, 'MSE': 1.9215965982288872, 'Rp': 0.99213641803824981}, {'R^2': 0.9848687744462441, 'MSE': 2.0631462094138278, 'Rp': 0.99240723684837506}, {'R^2': 0.9826962681204892, 'MSE': 2.3046327658833912, 'Rp': 0.99131797713106118}, {'R^2': 0.9717038213172116, 'MSE': 3.8069457416869978, 'Rp': 0.98576854479262532}, {'R^2': 0.9144033164783537, 'MSE': 12.338978574703111, 'Rp': 0.95644977430979328}, {'R^2': 0.6896759755903588, 'MSE': 43.86286187591211, 'Rp': 0.83964684541195078}, {'R^2': 0.6165582345787195, 'MSE': 53.872486653553686, 'Rp': 0.79351197595886847}], [{'R^2': 0.9877770007109786, 'MSE': 1.7341149624830037, 'Rp': 0.99387407895188518}, {'R^2': 0.9877434644679608, 'MSE': 2.0257998154382784, 'Rp': 0.99386293864198871}, {'R^2': 0.9856454965679888, 'MSE': 2.4552422800208049, 'Rp': 0.9928720102456059}, {'R^2': 0.9820502475916845, 'MSE': 3.1271765101933444, 'Rp': 0.99108242863910589}, {'R^2': 0.9791248063387743, 'MSE': 3.5507936646172964, 'Rp': 0.98950917483436296}, {'R^2': 0.8371185186684176, 'MSE': 25.832629826270125, 'Rp': 0.91738618729914667}, {'R^2': 0.6015215868966806, 'MSE': 67.776014033526522, 'Rp': 0.79588235459695877}, {'R^2': 0.5627401528652589, 'MSE': 74.500084396051719, 'Rp': 0.75824541605234475}], [{'R^2': 0.98785256823923, 'MSE': 1.5852187944677172, 'Rp': 0.99393929994423469}, {'R^2': 0.9874806971399123, 'MSE': 1.7851995664494729, 'Rp': 0.99372110116555012}, {'R^2': 0.979592533434873, 'MSE': 2.7574769710148486, 'Rp': 0.98981478518717048}, {'R^2': 0.981094923621148, 'MSE': 2.5208588484408518, 'Rp': 0.99053137334342123}, {'R^2': 0.9705041942799099, 'MSE': 4.196754229271809, 'Rp': 0.98520124042324853}, {'R^2': 0.9269398379740157, 'MSE': 9.701770288420704, 'Rp': 0.9630300670176356}, {'R^2': 0.7387828540913856, 'MSE': 36.063060575740764, 'Rp': 0.86179311922229396}, {'R^2': 0.5500018544573875, 'MSE': 63.194808163183126, 'Rp': 0.7525074353699569}],[{'R^2': 0.9860113481817915, 'MSE': 1.5025333990522498, 'Rp': 0.99312732899469258}, {'R^2': 0.9828304780316826, 'MSE': 1.9511340473567944, 'Rp': 0.99141355021046229}, {'R^2': 0.9805208326434397, 'MSE': 2.1219085892751073, 'Rp': 0.99025333576864594}, {'R^2': 0.9763273404793034, 'MSE': 2.8295787392803691, 'Rp': 0.98812434979732211}, {'R^2': 0.9699166887680121, 'MSE': 3.3732155867561509, 'Rp': 0.98488236446433564}, {'R^2': 0.7943590298970213, 'MSE': 24.666276628519597, 'Rp': 0.89607991679365562}, {'R^2': 0.6804586051171434, 'MSE': 36.671913568345786, 'Rp': 0.83061289196474541}, {'R^2': 0.5232508266338365, 'MSE': 57.237030901441081, 'Rp': 0.73404207486273731}]]


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
