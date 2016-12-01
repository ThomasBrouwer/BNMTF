"""
Plot the convergence of the many different NMF algorithms in a single graph,
with time on the x-axis. We make sure we rerun each method 10 times and take
the average timestamps.

We run our method on the entire random dataset, so no test set.

We use a dataset of I=100, J=80, K=10, with unit mean priors and zero mean unit
variance noise.

We have the following methods:
- VB NMF
- Gibbs NMF
- ICM NMF
- Non-probabilistic NMF
"""

import matplotlib.pyplot as plt, ast
metrics = ['MSE']#,'R^2','Rp']

MSE_max = 3
time_max = 2

folder = "./"

# VB NMF
vb_all_performances = eval(open(folder+'nmf_vb_performances.txt','r').read())
vb_all_times_average = ast.literal_eval(open(folder+'nmf_vb_times.txt','r').read())

# Gibbs NMF
gibbs_all_performances = eval(open(folder+'nmf_gibbs_performances.txt','r').read())
gibbs_all_times_average = ast.literal_eval(open(folder+'nmf_gibbs_times.txt','r').read())

# ICM NMF
icm_all_performances = eval(open(folder+'nmf_icm_performances.txt','r').read())
icm_all_times_average = ast.literal_eval(open(folder+'nmf_icm_times.txt','r').read())

# NP NMF
np_all_performances = eval(open(folder+'nmf_np_performances.txt','r').read())
np_all_times_average = ast.literal_eval(open(folder+'nmf_np_times.txt','r').read())



# Assemble the average performances and method names
methods = ['VB-NMF', 'G-NMF', 'ICM-NMF', 'NP-NMF']
all_performances = [
    vb_all_performances,
    gibbs_all_performances,
    icm_all_performances,
    np_all_performances
]
all_times = [
    vb_all_times_average,
    gibbs_all_times_average,
    icm_all_times_average,
    np_all_times_average
]
colours = ['r','b','g','c']

for metric in metrics:
    fig = plt.figure(figsize=(1.9,1.5))
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.17, top=0.95)
    #plt.title("Performances (%s) for different fractions of missing values" % metric)
    plt.xlabel("Time (seconds)", fontsize=8, labelpad=0)
    plt.ylabel(metric, fontsize=8, labelpad=-1)
    plt.yticks(range(0,MSE_max+1),fontsize=6)
    plt.xticks(fontsize=6)
    
    for method,performances,times,colour in zip(methods,all_performances,all_times,colours):
        x = times
        y = performances[metric]
        #plt.plot(x,y,label=method)
        plt.plot(x,y,linestyle='-', marker=None, label=method, c=colour)
    
    plt.xlim(0,time_max)
    if metric == 'MSE':
        plt.ylim(0,MSE_max)
    elif metric == 'R^2' or metric == 'Rp':
        plt.ylim(0,1)
        
    plt.savefig("../graphs_toy/mse_nmf_times.png", dpi=600)
    #plt.savefig("/home/tab43/Dropbox/Posters/Poster NIPS AABI 2016 v2/images/mse_nmf_times.png", dpi=1200)