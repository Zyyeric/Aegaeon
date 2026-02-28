# On head node
ray start --head --port=6789 --num-cpus=$(nproc --all) --resources='{"node_0": 1}'

# On second node
ray start --address=<head_ip>:6789 --num-cpus=$(nproc --all) --resources='{"node_1": 1}'

# Stopping the clusters (per node)
ray stop