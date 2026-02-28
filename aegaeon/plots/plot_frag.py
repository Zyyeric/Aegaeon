import json
import sys
import matplotlib.pyplot as plt
import os

fontsize = 20
fontdict = {'size': fontsize}

def compute(log_file):
    usages = []
    with open(log_file, 'r') as f:

        for line in f:
            if 'Block Usage' in line:
                usage = {}
            if '---------------------------------' in line:
                usages.append(usage)
            if 'cpu (shape' in line:
                sep = line.strip().split(' ')

                shape = ' '.join(sep[4:9])[:-1]
                slab = int(sep[9])
                free = int(sep[11])
                total = int(sep[13])
                usage[shape] = (slab, free, total)
    
    return usages


if __name__ == '__main__':
    log_file = '../logs/32-0.1.log'
    usages = compute(log_file)

    frag = {shape: (0, 0) for shape in usages[-1]}
    all_frag = (0, 0)

    for usage in usages:
        all_total_size = 0
        all_free_size = 0

        for shape, (slab, free, total) in usage.items():
            total_size = slab * (1024 ** 3)
            shape_tuple = eval(shape)
            free_size = free * shape_tuple[0] * shape_tuple[1] * shape_tuple[2] * shape_tuple[3] * shape_tuple[4] * 2
            
            if total_size > frag[shape][1]:
                frag[shape] = (free_size, total_size)
            all_total_size += total_size 
            all_free_size += free_size
        if all_total_size > all_frag[1]:
            all_frag = (all_free_size, all_total_size)
    
    labels = list(frag.keys())
    categories = ['Utilized', 'Fragmentation',]

    fig, ax = plt.subplots(figsize=(4, 3.6))

    colors = ['#4573c4', "#5CCCCF"]

    for j in range(len(labels)):
        frag_ratio = frag[labels[j]][0] / all_frag[1] * 100
        used_ratio = frag[labels[j]][1] / all_frag[1] * 100 - frag_ratio
        ax.bar(
            f'S{j}',
            used_ratio,
            bottom=0,
            color=colors[0],
            label=categories[0],
        )
        ax.bar(
            f'S{j}',
            frag_ratio,
            bottom=used_ratio,
            color=colors[1],
            label=categories[1],
        )
    
    all_frag_ratio = all_frag[0] / all_frag[1] * 100
    all_used_ratio = 100 - all_frag_ratio
    ax.bar(
        'All',
        all_used_ratio,
        bottom=0,
        color=colors[0],
        label=categories[0],
    )
    ax.bar(
       'All',
        all_frag_ratio,
        bottom=all_used_ratio,
        color=colors[1],
        label=categories[1],
    )

    # Add labels, title, and legend
    ax.set_xlabel('KV Cache Block Shape', fontdict=fontdict)
    ax.set_ylabel('Memory Usage (%)', fontdict=fontdict)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(ax.get_xticklabels(), fontdict=fontdict)
    ax.set_yticklabels(ax.get_yticks(), fontdict=fontdict)
    
    handles = [
        plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(2)
    ]

    ax.legend(
        handles,
        categories, 
        loc='upper left',
        ncol=1, 
        fontsize=18,
        handletextpad=0.2,
        columnspacing=0.5,
        frameon=True,
    )
    plt.ylim(0, 100)
    plt.tight_layout(pad=0.2)

    plt.savefig('frag.pdf')
