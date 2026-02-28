import json
import sys
import matplotlib.pyplot as plt
import os
import seaborn as sns

fontsize = 20
fontdict = {'size': fontsize}

if __name__ == '__main__':
    log_file = '../logs/32-0.1.log'

    with open(log_file, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if 'elapsed' in line]

    labels = ['7B', '9B', '13B']
    times = [[], [], []]

    for line in lines:
        sep = line.split(' ')
        model = int(sep[3][-2:])
        time = float(sep[-1][:-1])

        if model % 8 == 0:
            times[0].append(time)
        if model % 8 == 4:
            times[1].append(time)
        if model % 8 == 6:
            times[2].append(time)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(4, 3.6))
    
    for i, time in enumerate(times):
        sns.ecdfplot(time, label=labels[i], ax=ax)
    
    # Add labels, title, and legend
    ax.set_xlabel('Auto-Scale Time (s)', fontdict=fontdict)
    ax.set_ylabel('CDF', fontdict=fontdict)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize) 

    
    ax.legend(
        loc='lower right', 
        fontsize=18)
    plt.tight_layout(pad=0.2)

    plt.savefig('autoscale_cdf.pdf')
