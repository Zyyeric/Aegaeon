# Artifact Evaluation Guide

### Persisient storage

Xiang, Y. (2025). [Artifact] Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market. Zenodo. https://doi.org/10.5281/zenodo.16673199

### Prerequisite

* conda
* ubuntu 22.04
* 2x8 H100 nodes (for aegaeon); A100 80GB node (for muxserve); A10 node (for one of aegaeon's experiment)
    * The A100 node should have CUDA version 11.7/11.8
* ~256GB disk space (for storing model weights), ~512GB CPU memory

> **Note**: Experiments in the paper were conducted on 2x8 H800 nodes. We acknowledge that H800 might not be easily accessible, and have tested that using H100 produces reasonably close results. Other hardware (e.g., A100) are not guaranteed to yield the same end-to-end comparison.

### Getting Started

> Unzip the artifact content under `/root`; the rest of the scripts assumes this exact code structure.

Install `aegaeon`:
```bash
# Assume: under /root
cd aegaeon

# Install in conda; venv is also OK
conda create -n aegaeon python=3.10
pip install -e .

# Install the quick_model_loader; this requires Rust
cd quick_model_loader/
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustc --version

pip install -e .

# Verify installation
cd ..
python3 
>>> import quick_model_loader
>>> import aegaeon
```

Prepare models and dataset:
```bash
# Assume: under /root
mkdir models
cd models

apt-get install git-lfs
git lfs install

# For aegaeon
git lfs clone https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/
git lfs clone https://www.modelscope.cn/models/01ai/Yi-1.5-9B-Chat/
git lfs clone https://www.modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/

# For muxserve
git lfs clone https://huggingface.co/huggyllama/llama-7b
git lfs clone https://huggingface.co/huggyllama/llama-13b

# ShareGPT
cd ..
git lfs clone https://www.modelscope.cn/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split/
mv ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json ./

# By now, the dataset should be at /root/ShareGPT_V3_unfiltered_cleaned_split.json, 
# and the models should be under /root/models
```

Install `muxserve` (with our patch for AE): 
```bash
# Assume: under /root
# Adapted from https://github.com/hao-ai-lab/MuxServe

conda create -n muxserve python=3.8
conda activate muxserve
cd MuxServe-vLLM
pip install -e . 

cd ../MuxServe
pip install -e . 
pip install -r requirements.txt

pip install cvxpy # this was missing in the dependency

# Verify installation
python3
>>> import vllm
```

Install `ServerlessLLM` (with our patch for AE): 
```bash
# Assume: under /root
cd ServerlessLLM
git checkout aegaeon # make sure on the right branch

# Follow https://serverlessllm.github.io/docs/stable/deployment/single_machine/ (single machine, from source)
```

Prepare workloads:
```bash
# Assume: under /root, conda activate aegaeon
mkdir workloads
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.1 --duration 120
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.5 --duration 60
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.1 --inlen-scale 2.0 --duration 120
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.5 --inlen-scale 2.0 --duration 60
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.1 --outlen-scale 2.0 --duration 120
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.5 --outlen-scale 2.0 --duration 60
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.1 --outlen-scale 2.0 --duration 120
python3 aegaeon/aegaeon/workload.py --num-models $(seq 8 2 80) --arrival-rate 0.5 --outlen-scale 2.0 --duration 60
python3 aegaeon/aegaeon/workload.py --num-models 40 --arrival-rate $(seq 0.05 0.05 0.70)  --duration 60

ls workloads # expect files like "synth-8-0.1-1.0-1.0.csv"
```

Generate profile data for aegaeon (**optional**; all profile data for H100 is included already):
```bash
# Assume: under /root/aegaeon, conda activate aegaeon
python3 profiles/do_profile.py

# Paths and model names are set up for the "Qwen2.5-7B-Instruct", 
# "Yi-1.5-9B-Chat", and "Llama-2-13b-chat-ms" models only. 
# To profile for other models and under different TP (default 1), 
# adjust do_profile.py accordingly.
```
---

### Running Experiments

#### Figure 11 & 12 & 13 (end-to-end)

Run aegaeon:
> **Note**: Both of the H100 nodes should be configured according to the previous section, including pulling the model weights. Dataset is only needed on the head node.
```bash
# Assume: under /root/aegaeon, conda activate aegaeon

# On head node
ray start --head --port=6789 --num-cpus=$(nproc --all) --resources='{"node_0": 1}'

# On second node
ray start --address=<head_ip>:6789 --num-cpus=$(nproc --all) --resources='{"node_1": 1}'

# On head node
# (2-5 minutes per sample point on the plot; expect hours for this)
mkdir logs
mkdir -p plots/json

# rate=0.1
for model in $(seq 16 4 80); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.1.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.1;
done

# rate=0.5 
for model in $(seq 16 2 54); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.5.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.5;
done

# model=40
for rate in $(seq 0.05 0.05 0.70); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/40-$rate.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model 40 --arrival-rate=$rate;
done


# rate=0.1, ix2
for model in $(seq 16 4 80); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.1-ix2.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.1 --inlen-scale 2.0;
done

# rate=0.1, ox2
for model in $(seq 16 4 80); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.1-ox2.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.1 --outlen-scale 2.0;
done

# rate=0.5, ix2
for model in $(seq 16 2 54); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.5-ix2.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.5 --inlen-scale 2.0;
done

# rate=0.5, ox2
for model in $(seq 16 2 54); 
do 
    AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.5-ox2.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.5 --outlen-scale 2.0;
done

# rate=0.1, 0.5xSLO 
for model in $(seq 16 4 80); 
do 
    AEGAEON_TTFT_SLO=5 AEGAEON_TPOT_SLO=0.05 AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.1-0.5x.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.1;
done

# rate=0.1, 0.3xSLO 
for model in $(seq 16 4 80); 
do 
    AEGAEON_TTFT_SLO=3 AEGAEON_TPOT_SLO=0.03 AEGAEON_LOG_FILE="/root/aegaeon/logs/$model-0.1-0.3x.log" RAY_DEDUP_LOGS=0 python3 benchmark/benchmark_e2e.py --nnodes 2 --num-model $model --arrival-rate=0.1;
done

# Results should appear in /root/aegaeon/plots/json/
```

Run ServerlessLLM & ServerlessLLM+:
> **Note**: We implement ServerlessLLM+ in a simulator that is included in aegaeon. For a cleaner comparison, we run both ServerlessLLM and ServerlessLLM+ using simulation. 
> Running vanilla ServerlessLLM is possible (see later). However, due to implementation issues, it actually performs much worse than the simulation, and is highly unstable towards the higher load part of this experiment.
```bash
# Assume: under /root/aegaeon, conda activate aegaeon
cp -r /root/workloads/*.csv sim/simdata/ # only if workloads are updated

# rate=0.1
python3 -m sim.simulate --num-models $(seq 16 4 80) --arrival-rate 0.1

# rate=0.5
python3 -m sim.simulate --num-models $(seq 16 2 54) --arrival-rate 0.5

# model=40
python3 -m sim.simulate --num-models 40 --arrival-rate $(seq 0.05 0.05 0.70)

# rate=0.1, ix2
python3 -m sim.simulate --num-models $(seq 16 4 80) --arrival-rate 0.1 --inlen-scale 2.0

# rate=0.1, ox2
python3 -m sim.simulate --num-models $(seq 16 4 80) --arrival-rate 0.1 --outlen-scale 2.0

# rate=0.5
python3 -m sim.simulate --num-models $(seq 16 2 54) --arrival-rate 0.5 --inlen-scale 1.0

# rate=0.5
python3 -m sim.simulate --num-models $(seq 16 2 54) --arrival-rate 0.5 --outlen-scale 2.0

# Results should appear in /root/aegaeon/plots/json/
# Expect the SeLLM+ simulation to take significantly more time than SeLLM (up to hours)
```

Run muxserve:
```bash
# Assume: under /root/MuxServe, conda activate muxserve

bash scripts/start_mps.sh examples/basic/mps

# Prepare workloads
# NOTE: The first two commands should end with "MuxServe cannot find a placement for 18 models", 
# indicating that multiplexing with MuxServe cannot support > 16 models on an 8xA100 node, 
# or a maximum of 32 models on 2 nodes (see paper).
python3 benchmark/aegaeon/bench_end_to_end_muxserve.py --num-models 8 10 12 14 16 18 --arrival-rate 0.1 

python3 benchmark/aegaeon/bench_end_to_end_muxserve.py --num-models 8 10 12 14 16 18 --arrival-rate 0.5

python3 benchmark/aegaeon/bench_end_to_end_muxserve.py --num-models 8 10 12 14 16 --arrival-rate 0.1 0.5 --inlen-scale 2.0

python3 benchmark/aegaeon/bench_end_to_end_muxserve.py --num-models 8 10 12 14 16 --arrival-rate 0.1 0.5 --outlen-scale 2.0

# Replay workloads 
# Refer to https://github.com/hao-ai-lab/MuxServe/tree/main/benchmark/end_to_end/

# For example, for models = 8 and rate = 0.1
bash benchmark/aegaeon/run_end_to_end.sh muxserve 0,1,2,3 benchmark/aegaeon/model_cfgs/models8_rate0.1_scale_1.0_1.0/tmp_model_cfg_GPUnum8_mesh_size4_idx0.yaml  benchmark/aegaeon/workloads/models8_rate0.1_scale_1.0_1.0/sharegpt_n8_req.json
bash benchmark/aegaeon/run_end_to_end.sh muxserve 4,5,6,7 benchmark/aegaeon/model_cfgs/models8_rate0.1_scale_1.0_1.0/tmp_model_cfg_GPUnum8_mesh_size4_idx1.yaml  benchmark/aegaeon/workloads/models8_rate0.1_scale_1.0_1.0/sharegpt_n8_req.json

# Remember to stop MPS
bash scripts/stop_mps.sh examples/basic/mps

# We do not provide a script to collect the results;
# by viewing the summary after each run, it is clear that this setup corresponds to a light 
# load for the MuxServe system, and the SLO attainment is always 100%.
# However, MuxServe can only be deployed for up to 16 models per node (32 models for 2 nodes).
```

Plotting:
```bash
# Assume: under /root/aegaeon/plots, conda activate aegaeon
# Should have all the result files under /root/aegaeon/plots/json

# Produces 0.1.pdf, Figure 11(a)
python3 plot_e2e.py --spec 0.1 

# Produces 0.5.pdf, Figure 11(b)
python3 plot_e2e.py --spec 0.5

# Produces 40.pdf, Figure 11(c)
python3 plot_e2e.py --spec 40

# Produces 0.1-ix2.pdf, Figure 12(a)
python3 plot_e2e.py --spec 0.1-ix2

# Produces 0.1-ox2.pdf, Figure 12(b)
python3 plot_e2e.py --spec 0.1-ox2

# Produces 0.5-ix2.pdf, Figure 12(c)
python3 plot_e2e.py --spec 0.5-ix2

# Produces 0.5-ox2.pdf, Figure 12(d)
python3 plot_e2e.py --spec 0.5-ox2

# Produces slo0.5.pdf, Figure 13(a)
python3 plot_e2e.py --spec slo0.5

# Produces slo0.3.pdf, Figure 13(b)
python3 plot_e2e.py --spec slo0.3

```
> **What to Expect**: the resulting figures should show an all-round performance improvement for all systems (compared to ones in the paper), due to the switch from H800 to H100. The relative trend remains the same.


Run vanilla ServerlessLLM:
```bash
# Assume: under /root/ServerlessLLM
# Adapted from https://serverlessllm.github.io/docs/stable/deployment/single_machine/

# tmux window 0
conda activate sllm
ray start --head --port=6379 --num-cpus=$(nproc --all) --num-gpus=0 \
  --resources='{"control_node": 1000}' --block

# tmux window 1
conda activate sllm-worker
ray start --address=0.0.0.0:6379 --num-cpus=$(nproc --all)  --num-gpus=8 \
  --resources='{"worker_node": 1000, "worker_id_0": 1000}' --block

# tmux window 2
conda activate sllm-worker
sllm-store start --mem-pool-size 256GB --num-thread 16

# tmux window 3
conda activate sllm
ulimit -n 65536 # or as high as possible
sllm-serve start --enable-storage-aware

# tmux window 4
# Assume: under /root/ServerlessLLM/configs
conda activate sllm
./deploy 23
sllm-cli replay --workload /root/workloads/synth-16-0.1-1.0-1.0.csv # change 16 to other number of models
```
> This has multiple caveats:
> + During model deployment (`./deploy 23`), expect crashes (server internal error) every few models deployed. The deployed models will be cached under `/root/ServerlessLLM/models`, so this process should eventually finish.
> + In the case of crashing, re-launch the `sllm-serve` command. Expect having to restart the ray clusters on window 0 and 1 as well, should the re-launching fail.
> + Expect `sllm-serve` to crash due to "Too many open files". This is because ServerlessLLM depends on ray and it launches many ray actors, which use up sockets. The `ulimit -n` command helps, but does not solve the problem entirely. In fact, this number is tied with the number of models deployed, so expect higher failure rate with more models deployed.
> + Results of workload replay can be viewed at `/root/ServerlessLLM/configs/latency_results.json`. Expect exceedingly high TTFTs (up to tens of seconds), which are much worse than in our simulation above.
> + Expect overloading with lots of models (e.g., with `synth-24-0.1-1.0-1.0.csv`), where no progress is made. The logs on window3 should report "not enough GPUs".

#### Figure 14 & 15 & 16 (sensitive analysis)

```bash
# Assume: under /root/aegaeon/plots, conda activate aegaeon
# These use results from the end-to-end runs

# Dependency for plotting
pip install seaborn

# Produces latency_breakdown.pdf, Figure 14
python3 plot_breakdown.py

# Produces autoscale_cdf.pdf, Figure 15(a)
python3 plot_autoscale_cdf.py

# Produces kvcache_cdf.pdf, Figure 15(b)
python3 plot_kvcache_cdf.py

# Produces frag.pdf, Figure 16
python3 plot_frag.py
```
> **What to Expect**: Due to the shift from H800 to H100 and the randomness of relying on logs, expect some differences in the figures.


#### Figure 17 (A10/TP4)

Walkthrough for the A10 figure:
* Install aegaeon on an A10 node, according to Getting Started
* Get Yi1.5-6B-Chat and Qwen2.5-7B model weights; update aegaeon/models.py accordingly
* Generate profiling data (see Getting Started)
* Run aegaeon with num-models=[4, 5, 6, 7, 8, 9, 10], arrival-rate=0.1, obtain the json files
* Plot with the following script

Alternatively, validate using the existing results:
```bash
# Data in /root/aegaeon/plots/a10/
python3 plot_a10.py
```

Walkthrough for the TP4 figure:
* Get Qwen2.5-72B model weights;  update aegaeon/models.py accordingly
* Generate profiling data for TP4 (see Getting Started)
* Run aegaeon with num-models=4, arrival-rate=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], obtain the json files
* Plot with the following script

Alternatively, validate using the existing results:
```bash
# Data in /root/aegaeon/plots/tp4/
python3 plot_tp4.py
```
---

### Common Issues

* "GLIBCXX_3.4.32 not found"?

This is a GCC version issue. Try this:
```bash
conda install -c conda-forge libstdcxx-ng
```
Alternatively, one of [these solutions](https://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error) should work.


* Error during aegaeon runs (regarding `hidden_states`, GPU blocks, or CPU blocks)?

This can occur under high load due to system lag. Try rerunning the error case a few times. 
If the error has to do with "not enough CPU blocks", try the following:
```bash
rm /dev/shm/aegaeon_cpu_cache
# Rerun with --cpu-num-slabs 384 (or other values greater than 256)
```

* Error installing muxserve?

MuxServe uses a modified version of vllm 0.2.0, which depends on torch 2.0.1 with wheels up to only CUDA 11.8. Deploying MuxServe on H100 clusters (CUDA >= 12) is unlikely to succeed (compiling torch from source may work, but this is not tested). With A100 clusters, make sure CUDA version is 11.7 or 11.8.
