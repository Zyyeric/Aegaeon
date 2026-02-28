import json
import sys
import matplotlib.pyplot as plt
import os

fontsize = 20
fontdict = {'size': fontsize}

def compute(json_file):

    with open(json_file, 'r') as f:
        req_metrics = json.load(f)
    
    prefill_wait_time = 0
    prefill_time = 0
    decode_wait_time = 0
    decode_time = 0
    control_overhead = 0
    block_overhead = 0
    for req_id, metric in req_metrics.items():
        prefill_wait_time += metric["prefill_wait_time"]
        prefill_time += metric["prefill_time"]
        decode_wait_time += metric["decode_wait_time"]
        decode_time += metric["decode_time"]
        control_overhead += metric["control_overhead"]
        block_overhead += metric["block_overhead"]
    
    all_time = (
        prefill_wait_time + prefill_time + decode_wait_time + decode_time 
        + control_overhead + block_overhead
    )

    return [
        prefill_wait_time/all_time*100,
        prefill_time/all_time*100,
        decode_wait_time/all_time*100,
        decode_time/all_time*100,
        control_overhead/all_time*100,
        block_overhead/all_time*100,
    ]

if __name__ == '__main__':
    json_files = [
        'json/run-16-0.1-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-32-0.1-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-64-0.1-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-16-0.5-1.0-1.0-ttft10.0-tpot0.1.json',
        'json/run-32-0.5-1.0-1.0-ttft10.0-tpot0.1.json',
    ]
    
    times = [compute(file) for file in json_files]

    categories = ['Prefill Waiting', 'Prefill Execution', 'Decoding Waiting',
                  'Decoding Execution', 'Control Overhead', 'Data Overhead']
    colors = [
        (51/255, 138/255, 114/255),
        (156/255, 168/255, 63/255),
        '#4573c4',
        "#5CCCCF",
        (229/255, 216/255, 187/255),
        (21/255, 26/255, 46/255),
    ]
    labels = ['16x0.1', '32x0.1', '64x0.1', '16x0.5', '32x0.5']

    fig, ax = plt.subplots(figsize=(5.95, 3.6))

    bottoms = [0] * 5
    for i in range(6):
        for j in range(5):
            ax.bar(
                labels[j],
                times[j][i], 
                bottom=bottoms[j], 
                color=colors[i], 
                label=categories[i],
            )
       
            bottoms[j] += times[j][i]
    
    # Add labels, title, and legend
    ax.set_xlabel('Setup (#model x RPS)', fontdict=fontdict)
    ax.set_ylabel('Latency Breakdown (%)', fontdict=fontdict)
    # ax.set_xticks([0,1,2,3,4], labels=['16x0.1', '32x0.1', '64x0.1', '16x0.5', '32x0.5'])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=fontdict)
    ax.set_yticklabels(ax.get_yticks(), fontdict=fontdict)
    
    handles = [
        plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(6)
    ]
    ax.legend(
        handles,
        categories, 
        loc='upper center', 
        ncol=2, 
        bbox_to_anchor=(0.45, 1.8), 
        handletextpad=0.2,
        columnspacing=0.5,
        fontsize=18,
        frameon=False
    )
    plt.ylim(0, 100)
    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(top=0.65)

    plt.savefig('latency_breakdown.pdf')
