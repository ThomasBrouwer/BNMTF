"""
Plot the performances of the many different NMTF algorithms in a single bar chart.

We plot the average performance across all 10 attempts for different noise levels.

We have the following methods:
- VB NMTF
- Gibbs NMTF
- ICM NMTF
- Non-probabilistic NMTF
"""

import matplotlib.pyplot as plt, numpy
metrics = ['MSE'] #['MSE','R^2','Rp']

noise_ratios = [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ]
# Which noise ratios we put in the bar chart:
shown_noise_ratios = [ 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 ] #[ 0, 0.01, 0.02, 0.05, 0.1 ]
indices_selected = [noise_ratios.index(noise) for noise in shown_noise_ratios]


# Settings for the bar chart
N = len(shown_noise_ratios) # number of bars
ind = numpy.arange(N) # x locations groups
width = 0.2 # width of bars
MSE_max = 30


# VB NMTF

# Gibbs NMTF

# ICM NMTF

# Non-probabilistic NMTF


# Assemble the average performances and method names
methods = []#['VB-NMTF', 'G-NMTF', 'ICM-NMTF', 'NP-NMTF']
avr_performances = [
    #vb_average_performances,
    #gibbs_average_performances,
    #icm_average_performances,
    #np_average_performances
]
colours = ['r','b','g','c']

for metric in metrics:
    plt.figure()
    #plt.title("Performances (%s) for different noise ratios" % metric)
    plt.xlabel("Noise to signal ratio", fontsize=16)
    plt.ylabel(metric, fontsize=16)
    
    x = noise_ratios 
    offset = 0
    for (method, avr_performance, colour) in zip(methods,avr_performances,colours):
        all_performances = avr_performance[metric]
        y = numpy.array([all_performances[i] for i in indices_selected])
        plt.bar(ind+offset, y, width, label=method, color=colour)
        offset += width
        
    plt.ylim(0,MSE_max)
    plt.xticks(numpy.arange(N) + 2*width, x)
    
    plt.legend(loc=0) 
    
    