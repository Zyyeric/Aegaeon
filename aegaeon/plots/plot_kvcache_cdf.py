import json
import sys
import matplotlib.pyplot as plt
import os

fontsize = 20
fontdict = {'size': fontsize}


def compute(json_file):

    with open(json_file, 'r') as f:
        req_metrics = json.load(f)
    
    block_overheads = []
    for req_id, metric in req_metrics.items():
        block_overheads.append(metric["control_overhead"] + metric["block_overhead"])

    # print(max(block_overheads))
    return block_overheads

if __name__ == '__main__':
    json_files = [
        'json/run-16-0.1-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-32-0.1-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-64-0.1-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-16-0.5-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-32-0.5-1.0-1.0-ttft10.0-tpot0.1.json',
    ]
    
    times = [compute(file) for file in json_files]
    labels = ['16x0.1', '32x0.1', '64x0.1', '16x0.5', '32x0.5']

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(4, 3.6))

    for i in range(5):
        ax.ecdf(times[i], label=labels[i])
    
    # Add labels, title, and legend
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_xlabel('KV Cache Sync (s)', fontdict=fontdict)
    ax.set_ylabel('CDF', fontdict=fontdict)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize) 

    ax.legend(
        loc='lower right', 
        fontsize=18)
    plt.tight_layout(pad=0.2)

    plt.savefig('kvcache_cdf.pdf')
