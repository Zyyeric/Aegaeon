from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import json 
import sys
import os

def compute(json_file, ttft_slo, tpot_slo):
    
    # Plot SLO Attainment
    TTFT_SLO = ttft_slo
    TPOT_SLO = tpot_slo

    with open(json_file, 'r') as f:
        req_metrics = json.load(f)
    
    qos_list = []
    for req_id, metrics in req_metrics.items():
        # if per_token_ms[0] == -1:
        #     qos_list.append(0)
        #     continue

        per_token = metrics["per_token"]
        
        qos = 0
        start = 0
        accum = 0
        target = TTFT_SLO
        sum_token = 0
        for token in per_token:
            sum_token += token
            if sum_token - start <= target:
                accum = target
                target += TPOT_SLO
            else:
                qos += accum
                start = sum_token
                accum = 0
                target += TPOT_SLO
        qos += min(target, sum(per_token)-start)
        qos_list.append(qos / sum(per_token))

    return(sum(qos_list)/len(qos_list))


fig, ax = plt.subplots(figsize=(5, 3.2))

labels = ["Normal", "Strict", "Loose"]
lines = ["o-", "v-", "^-"]
fontsize = 22
fontdict = {'size': fontsize}

def plot_one(ax: Axes, json_files, ttft_slo, tpot_slo, label, line):
    
    qos = [compute(f'a10/{json_file}', ttft_slo, tpot_slo) for json_file in json_files]

    ax.set_ylabel('SLO Attainment (%)', fontdict=fontdict, loc='top')
    ax.margins(x=0, y=0)
    ax.grid(False)

    ax.set_xlabel('#Models', fontdict=fontdict)
    ax.set_xlim(3.5, 10.5)
    ax.set_xticks([4, 6, 8, 10])
    ax.set_xticklabels(ax.get_xticks(), fontdict=fontdict)

    ax.set_ylim(50, 105)
    ax.set_yticks([50, 75, 100])
    ax.set_yticklabels(ax.get_yticks(), fontdict=fontdict)

    X = [i for i in range(4, 11)]
    yvals = [100 * v for v in qos]
    ax.plot(X, yvals, line, label=label, color='tab:blue', linewidth=2)

json_files = [f'{i}-0.1.json' for i in range(4, 11)]
outfile = f'a10.pdf'
plot_one(ax, json_files, 5, 0.1, 'Normal', lines[0])
plot_one(ax, json_files, 5, 0.05, 'Strict', lines[1])
plot_one(ax, json_files, 5, 0.2, 'Loose', lines[2])
ax.axhline(y=90, color='tab:gray', linestyle='--', linewidth=1)

fig.legend(labels, loc='lower center', bbox_to_anchor=(0.44, 0.2), fontsize=fontsize-2)
plt.tight_layout(pad=0.2)

plt.savefig(outfile)

