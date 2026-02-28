# README
NOTE(wyt): Currently we use cmdline python3.10 and pip3.10, instead of python3 and pip3.

0. On jump machine:
Choose your own PREFIX! my PREFIX is tomqi.

NOTE(tomqi): rsync is disabled when using ceph as sync tool. Do make sure you have sglang and nixl available in your ceph mount point.

```bash
export PREFIX=tomqi
mkdir ~/${PREFIX}-sgl-workspace/ && cd ~/${PREFIX}-sgl-workspace
# NOTE: make sure the dirname matches your ceph mount point.
git clone -b cp git@github.com:TomQuartz/sglang.git sglang-dev
cd sglang-dev
```

1. Run below commands on every gemini hosts:
NOTE(wyt): Do it multiple time if you use gemini tools, make sure damn 15s timeout will not happen.
```bash
  export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113
  export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113
  export http_proxy=http://hk-mmhttpproxy.woa.com:11113
  export https_proxy=http://hk-mmhttpproxy.woa.com:11113
  export NO_PROXY=127.0.0.1,0.0.0.0
  apt update
  apt install -y openssh-server
  echo -e "PermitRootLogin yes\nPasswordAuthentication yes\nPort 2222" | sudo tee -a /etc/ssh/sshd_config > /dev/null
  sudo service ssh restart
  echo "root:123456wyt" | chpasswd
```

2. Configure `~/.ssh/config` on jump machine, like below:
The prefix "tomqi" is important. We will use it later.

```shell
cp ~/.ssh/config ~/.ssh/config.bak
sed -i '/^Host tomqi[0-3]/,/^$/d' ~/.ssh/config

cat >> ~/.ssh/config << EOF
Host tomqi0
  Hostname 29.225.241.12
  User root
  Port 2222
  StrictHostKeyChecking no
Host tomqi1
  Hostname 29.225.241.184
  User root
  Port 2222
  StrictHostKeyChecking no
Host tomqi2
  Hostname 29.225.241.154
  User root
  Port 2222
  StrictHostKeyChecking no
Host tomqi3
  Hostname 29.225.241.104
  User root
  Port 2222
  StrictHostKeyChecking no
EOF
```

3. Run on jump machine:
  - `PREFIX=tomqi bash jump-initialize-ceph.sh ssh`
  - `PREFIX=tomqi bash jump-initialize-ceph.sh sgl`
  This script will do lots of things. Check it please.
  Use your own PREFIX!
4. On jump machine, make sure the following things in `$SGL_MNT_POINT/pd-test/pdconfig.py` meet your expectation:
  - PREFIX
  - PD configuration
  - MODEL
  - DATASET
  - Batch size
  - ...

NOTE(tomqi): If using ceph, develop in your ceph mount point.

5. Enjoy!

```bash
ssh tomqi0
cd $SGL_MNT_POINT/pd-test/
export PREFIX=tomqi
conda activate $PREFIX
export BENCH_PYTHON_CMD=$(conda run -n $PREFIX which python3.10)
# R1 1P1D tp8 dp1
python3.10 pd.py launch --plan r1 -p 1 -d 1 --tp-size 8 --mem-fraction-static 0.9 --max-running-requests 1
TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 0.5 -n 300 --max-tokens 1
# R1 3P1D tp16 dp16
python3.10 pd.py launch --plan r1 -p 3 -d 1 --tp-size 16 --dp-attn --dp-size 16 --max-running-requests 160
DP_ATTN_LOGS=1 TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 4 -n 200 480
# QwQ 4P1D tp4
python3.10 pd.py launch --plan qwq --max-running-requests 128
TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 4 -n 200 480
# Clean up
python3.10 pd.py clean
# ps -ef | grep 'sglang' | grep -v grep | grep -v defunct | awk '{print $2}' | xargs -r kill -SIGKILL; ps aux | grep 'sglang' | grep -v defunct; pkill -f sglang
```

Check benchmark result and sglang logs on `tomqi0:/tmp/`


6. Profiling
(echoxiang) To profile the instances,
```bash
export PREFIX=echoxiang
conda activate $PREFIX
export BENCH_PYTHON_CMD=$(conda run -n $PREFIX which python3.10)
export MNT_POINT="/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_$PREFIX/"
cd $MNT_POINT/sglang-dev/pd-test/

python3.10 pd.py launch --plan r1 -p 1 -d 1 --tp-size 8 --mem-fraction-static 0.9 --max-running-requests 1 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 4096 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 8192 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 12288 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 16384 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 20480 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 24576 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 28672 --profile

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 2 8 --max-tokens 1 --dummy-prompt-len 32768 --profile

```

7. Non-PD-disaggregated case (HACK around the v0.4.7 bug for now)

All on node 0:

```bash
# Deactivate conda env and don't be under pd-test/; the profile.py file confuses Python
cd $MNT_POINT/sglang-dev

PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 1 --node-rank 0 --dist-init-addr "$PREFIX"0:42993 --tp 8 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 2 --port 29921 ) 2>&1 | tee tp8.log

# Change LOGNAME
/root/miniconda3/envs/echoxiang/bin/python3.10 /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/sglang-dev/pd-test/benchmark-openai-zzi-conductor-wxg.py --d DummyDataset --f /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/qa_out_formatted.jsonl --o /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/logs/tfcc_benchmark_LOGNAME.log --t /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --n 2 --qps 1.00 --max-tokens 1 --random --host "$PREFIX"0 --port 29921 --model default --prompt-len 4096

```

TP16:

```bash
PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 2 --node-rank 0 --dist-init-addr "$PREFIX"0:42993 --tp 16 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 1 --port 29921 ) 2>&1 | tee tp16.log

PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 2 --node-rank 1 --dist-init-addr "$PREFIX"0:42993 --tp 16 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 1 --port 29921 ) 2>&1 | tee tp16.log
```

8. CP

```bash
export PREFIX=tomqi
export BENCH_PYTHON_CMD=$(conda run -n $PREFIX which python3.10)
export MNT_POINT="/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_$PREFIX/"
cd $MNT_POINT/sglang-dev/

PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/"$PREFIX"/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 4 --node-rank 0 --dist-init-addr "$PREFIX"0:42993 --tp-size 8 --cp-size 4 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 1 --port 29921 ) 2>&1 | tee cp0.log

PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/"$PREFIX"/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 4 --node-rank 1 --dist-init-addr "$PREFIX"0:42993 --tp-size 8 --cp-size 4 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 1 --port 29921 ) 2>&1 | tee cp1.log

PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/"$PREFIX"/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 4 --node-rank 2 --dist-init-addr "$PREFIX"0:42993 --tp-size 8 --cp-size 4 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 1 --port 29921 ) 2>&1 | tee cp2.log

PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_TORCH_PROFILER_DIR=/tmp/sprofile /root/miniconda3/envs/"$PREFIX"/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 4 --node-rank 3 --dist-init-addr "$PREFIX"0:42993 --tp-size 8 --cp-size 4 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --disable-cuda-graph --chunked-prefill-size 45056  --log-level debug --enable-metrics --max-running-requests 1 --port 29921 ) 2>&1 | tee cp3.log

# Benchmark on node 0
/root/miniconda3/envs/"$PREFIX"/bin/python3.10 /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/sglang-dev/pd-test/benchmark-openai-zzi-conductor-wxg.py --d WeChatSearchQaOutFormatted --f /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/qa_out_formatted.jsonl --o /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/logs/tfcc_benchmark_cp2.log --t /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --n 2 --qps 1.00 --max-tokens 32 --host "$PREFIX"0 --port 29921 --model default
```

9. CP benchmark

```bash
export PREFIX=tomqi
export BENCH_PYTHON_CMD=$(conda run -n $PREFIX which python3.10)

# install openai_benchmark
/root/miniconda3/envs/"$PREFIX"/bin/python3.10 -m pip install -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com libra-openai-benchmark

# Benchmark on node 0
/root/miniconda3/envs/"$PREFIX"/bin/python3.10 -m openai_benchmark.benchmark_serving --model default --host "$PREFIX"0 --port 29921 --endpoint /v1/chat/completions --dataset-name jsonl --dataset-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/qa_out_formatted.jsonl --backend openai-chat --tokenizer /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1   --result-dir /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/cp-prefill/bench_e2e --add-uuid --metric-percentiles 90,99,100 --percentile-metrics ttft,tpot,itl,e2el --save-result --num-prompts 600 --max-concurrency 1 --jsonl-output-len 1 # --profile
```

### TP8
========== Serving Benchmark All Result ==========
Successful requests:                     300       
Benchmark duration (s):                  495.21    
Total input tokens:                      4264696   
Total generated tokens:                  300       
Request throughput (req/s):              0.61      
Output token throughput (tok/s):         0.61      
Total Token throughput (tok/s):          8612.46   
Tokens per iteration:                    0.50      
---------------Time to First Token----------------
Mean TTFT (ms):                          1650.09   
Median TTFT (ms):                        1351.54   
P90 TTFT (ms):                           3302.53   
P99 TTFT (ms):                           4087.66   
P100 TTFT (ms):                          4195.74 

### TP16
========== Serving Benchmark All Result ==========
Successful requests:                     300       
Benchmark duration (s):                  367.24    
Total input tokens:                      4264724   
Total generated tokens:                  300       
Request throughput (req/s):              0.82      
Output token throughput (tok/s):         0.82      
Total Token throughput (tok/s):          11613.81  
Tokens per iteration:                    0.50      
---------------Time to First Token----------------
Mean TTFT (ms):                          1223.49   
Median TTFT (ms):                        1050.64   
P90 TTFT (ms):                           2323.30   
P99 TTFT (ms):                           2773.61   
P100 TTFT (ms):                          2836.81   


### CP2

======== Serving Benchmark Steady Result =========
Successful requests:                     298
Benchmark duration (s):                  288.37
Total input tokens:                      4237365
Total generated tokens:                  298
Request throughput (req/s):              1.03
Output token throughput (tok/s):         1.03
Total Token throughput (tok/s):          14695.23
Tokens per iteration:                    0.50
---------------Time to First Token----------------
Mean TTFT (ms):                          967.07
Median TTFT (ms):                        810.18
P90 TTFT (ms):                           1914.93
P99 TTFT (ms):                           2323.30
P100 TTFT (ms):                          2353.35

### CP3

======== Serving Benchmark Steady Result =========
Successful requests:                     298
Benchmark duration (s):                  218.44
Total input tokens:                      4237325
Total generated tokens:                  298
Request throughput (req/s):              1.36
Output token throughput (tok/s):         1.36
Total Token throughput (tok/s):          19399.78
Tokens per iteration:                    0.50
---------------Time to First Token----------------
Mean TTFT (ms):                          732.42
Median TTFT (ms):                        617.05
P90 TTFT (ms):                           1396.60
P99 TTFT (ms):                           1688.07
P100 TTFT (ms):                          1700.28

### CP4

Successful requests:                     298
Benchmark duration (s):                  180.74
Total input tokens:                      4237296
Total generated tokens:                  298
Request throughput (req/s):              1.65
Output token throughput (tok/s):         1.65
Total Token throughput (tok/s):          23445.33
Tokens per iteration:                    0.50
---------------Time to First Token----------------
Mean TTFT (ms):                          605.92
Median TTFT (ms):                        509.37
P90 TTFT (ms):                           1166.03
P99 TTFT (ms):                           1350.36
P100 TTFT (ms):                          1405.75


10. CP with PD disaggregation (echoxiang)

```bash
python3.10 pd.py launch --plan r1 -p 1 -d 1 --tp-size 8 --cp-size 2 --mem-fraction-static 0.9 --max-running-requests 1

TFCC_BENCHMARK=1 python3.10 pd.py bench --qps 1 -n 8 --max-tokens 32
```

Alternatively, use the raw commands for easier debugging:
```bash
PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 2 --node-rank 0 --dist-init-addr echoxiang0:31297 --tp 8 --cp-size 2 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --chunked-prefill-size 45056  --log-level debug --enable-metrics --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 19875 --max-running-requests 1 --port 50733 2>&1 | tee p0.log ) 


PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.launch_server --nnodes 2 --node-rank 1 --dist-init-addr echoxiang0:31297 --tp 8 --cp-size 2 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --chunked-prefill-size 45056  --log-level debug --enable-metrics --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 19875 --max-running-requests 1 --port 50205 2>&1 | tee p1.log) 



PS1=[] source ~/.bashrc  && ( UCX_TLS=rc,gdr_copy,rc_x,cuda_copy,cuda_ipc UCX_NET_DEVICES=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1 UCX_LOG_LEVEL=info NCCL_DEBUG=WARN SGLANG_PD_NIXL_DEBUG_TRANSFER_TIME=1 SGL_ENABLE_JIT_DEEPGEMM=0 /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.launch_server --host 0.0.0.0 --nnodes 1 --node-rank 0 --dist-init-addr echoxiang2:13251 --tp 8 --model-path /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --trust-remote-code --disable-radix-cache --schedule-policy fcfs --mem-fraction-static 0.9 --disable-overlap-schedule --chunked-prefill-size 45056  --log-level debug --enable-metrics --disaggregation-mode decode --disaggregation-transfer-backend nixl --max-running-requests 1 --port 44095 2>&1 | tee d.log )  



PS1=[] source ~/.bashrc  && ( /root/miniconda3/envs/echoxiang/bin/python3.10 -m sglang.srt.disaggregation.mini_lb --prefill http://echoxiang0:50733 --decode http://echoxiang2:44095 --host 0.0.0.0 --port 17423 --prefill-bootstrap-ports 19875 2>&1 | tee lb.log)  


/root/miniconda3/envs/"$PREFIX"/bin/python3.10 /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/sglang-dev/pd-test/benchmark-openai-zzi-conductor-wxg.py --d WeChatSearchQaOutFormatted --f /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/qa_out_formatted.jsonl --o /mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_"$PREFIX"/tfcc_pd/benchmark/logs/tfcc_benchmark_cp+pd.log --t /mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/DeepSeek-R1_with_draft/DeepSeek-R1 --n 1 --qps 1.00 --max-tokens 32 --host "$PREFIX"0 --port 17423 --model default 2>&1 | tee bench.log
```


TODO: CP with PD-dist benchmark