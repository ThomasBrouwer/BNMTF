"""
Test the performance of Gibbs sampling for recovering a toy dataset, where we
vary the level of noise.
We repeat this 10 times per fraction and average that.

We use the correct number of latent factors and flatter priors than used to generate the data.

I, J, K, L = 100, 80, 5, 5

The noise levels indicate the percentage of noise, compared to the amount of 
variance in the dataset - i.e. the inverse of the Signal to Noise ratio:
    SNR = std_signal^2 / std_noise^2
    noise = 1 / SNR
We test it for 1%, 2%, 5%, 10%, 20%, 50% noise.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../"
sys.path.append(project_location)

from BNMTF.code.models.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.data_toy.bnmtf.generate_bnmtf import add_noise, try_generate_M
from BNMTF.code.cross_validation.mask import calc_inverse_M

import numpy, matplotlib.pyplot as plt

##########

fraction_unknown = 0.1
noise_ratios = [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ] # 1/SNR

input_folder = project_location+"BNMTF/data_toy/bnmtf/"

repeats = 10
iterations = 2000
burn_in = 1800
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
noise_ratios = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] 
all_performances = {'R^2': [[0.9999640982474265, 0.999976662301363, 0.9999843514925597, 0.9999691931995058, 0.9999837090405099, 0.9999810598672735, 0.9999794549577409, 0.9999757021424392, 0.9963706851626953, 0.9999774424439364], [0.9887188254670738, 0.9876487813733615, 0.9888061298199565, 0.9883583710771858, 0.9896117991659635, 0.9861289512300546, 0.9872263835753372, 0.990472931494721, 0.9897171806347015, 0.9894613944026888], [0.9824560689479832, 0.9776955917526902, 0.9797947830345879, 0.9759832577953403, 0.9826032048727562, 0.9745347061180676, 0.9825738662186562, 0.9751853238589827, 0.97544680734672, 0.9786223927141903], [0.9527822862881948, 0.9407587376787945, 0.9430675459250004, 0.9427957003932831, 0.9482435559002408, 0.9479445529988814, 0.9434689531211573, 0.9529238496238643, 0.9472045133847624, 0.9536730490771194], [0.8865216326992937, 0.8870854653801203, 0.8880645484076666, 0.8921851145355723, 0.9125682113099509, 0.903454618164275, 0.9159655611024895, 0.888532025977158, 0.897404427655667, 0.8910352093577653], [0.7930249179029341, 0.7760057578624064, 0.8128742622786951, 0.8250443649828061, 0.810429706755466, 0.8100275399079921, 0.7866149458605467, 0.7879558264299412, 0.8403259813883655, 0.8224473824475729], [0.6838060862552677, 0.6777566749104094, 0.6236466849385653, 0.6416699811809343, 0.6521527397996659, 0.6577822072739117, 0.6300743802083444, 0.675574535888237, 0.6290401848023921, 0.6393660563558972]], 'MSE': [[0.016447594702847932, 0.01386835136476525, 0.0084577958536406466, 0.016290681282664338, 0.0089278291804378261, 0.011778688571337022, 0.012845321662760647, 0.012975383723015306, 1.9299812259530333, 0.0099777390138387281], [6.5367351673391418, 6.9483066510321816, 6.336986548645025, 7.06885439421072, 6.1376115329635139, 6.296995102228899, 6.6836440385620017, 6.972918283415285, 6.3626498685059971, 6.0509739453395976], [11.364130312368307, 14.211802881155956, 13.260938146974997, 13.659130540957248, 12.883066955577329, 13.006480902682794, 13.888108548803459, 12.63665697101108, 14.472569318504382, 13.134452715011811], [30.697442134390943, 31.104435533450637, 31.170023684032106, 31.644856938847866, 30.672401300938727, 30.27810278529028, 31.515588612131832, 33.105725119101628, 29.862474846697754, 30.668839914644238], [63.561091627935696, 61.147254483535768, 69.28178718762409, 61.571713640126248, 63.596114873964233, 68.828675169762121, 64.004825226366648, 64.971258277011387, 61.892934315310342, 64.637825688920799], [134.30557557257518, 130.9282044804043, 120.59565632726452, 121.38654723178968, 134.45885036015062, 128.62771049553595, 132.09493606715256, 128.71505507671648, 128.86432064560387, 116.39656612591978], [298.51302322400056, 282.31148793814913, 297.82639537004036, 323.85471704384696, 306.68057654971426, 254.92084483757887, 307.52553881157149, 298.9611220320088, 282.3800179878333, 313.5240999585651]], 'Rp': [[0.99998206058877015, 0.99998837465016099, 0.99999224250650387, 0.99998460020604119, 0.99999189792924048, 0.99999058076471636, 0.99998995780983757, 0.99998793402091002, 0.99821694849476239, 0.99998872460645849], [0.99436522659988136, 0.99387216979092929, 0.99439760221687701, 0.99417485433845865, 0.99482786301751724, 0.99321251979179193, 0.99359320240177307, 0.99524250177724449, 0.99484955447092305, 0.99473989564682297], [0.99119067944947858, 0.9888332231255188, 0.98993393512796157, 0.98803441729420349, 0.99135150041135078, 0.98718813011441842, 0.99126450210607919, 0.98764843448375095, 0.98779838377761142, 0.98931841924626596], [0.97623024098311173, 0.97086039617202002, 0.97126046185367154, 0.97128119730708184, 0.97388725669376819, 0.97374455971947804, 0.97181423342449469, 0.976360007343263, 0.9733636172290997, 0.97660586888208045], [0.9419407598203724, 0.94202470057602605, 0.94310303819095465, 0.9447641052649135, 0.95584821964837408, 0.95078050084487853, 0.95774460960621355, 0.94293660844736382, 0.94767227775898821, 0.94438868683755117], [0.89240304983805274, 0.88270564457583844, 0.90168222038427259, 0.90836019968339343, 0.90033342108224834, 0.90193109192403353, 0.88691816036256255, 0.88823604262690059, 0.9178699846639875, 0.90700771250264534], [0.82775022906788132, 0.82428462381119383, 0.78997529777273479, 0.80139742364172561, 0.80952936879035087, 0.81295453109753724, 0.79415852587625946, 0.82272736783784417, 0.7936576640461962, 0.79975504303526301]]} 
average_performances = {'R^2': [0.99961623588554505, 0.98861507482410427, 0.97848960026599729, 0.94728627443912983, 0.89628168145899578, 0.80647506858167262, 0.65108695316136256], 'MSE': [0.20415506113083409, 6.5395675532242352, 13.251733729304737, 31.071989086952605, 64.34934804905572, 127.63734223831129, 296.64978237533086], 'Rp': [0.99981133215774032, 0.99432753900522197, 0.98925616251366399, 0.9735407839608069, 0.94712035069956357, 0.89874475276439347, 0.80761900749769866]}
'''


# Plot the MSE, R^2 and Rp
for metric in metrics:
    plt.figure()
    x = noise_ratios
    y = average_performances[metric]
    plt.plot(x,y)
    plt.xlabel("Noise ratios missing")
    plt.ylabel(metric)