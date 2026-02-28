import os
import torch
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from aegaeon.utils import DeviceType

DIR = os.path.dirname(__file__)

# Number of GPUs
N = torch.cuda.device_count()
DEVICE_TYPE = DeviceType.from_str(torch.cuda.get_device_name(0))

# Per-model cases
CASES = [
    (ilen, bs)
    for ilen in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for bs in ([1, 2, 4, 8, 16] + list(range(30, 210, 10)))
    if ilen * bs <= 8192 and (ilen + 10) * bs < 10240
] 
# CASES = [
#     (16, 1)
# ] 

# Models and workloads
QWEN_7B = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Qwen-7B/", "qwen_7b")
QWEN_7B_CHAT = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Qwen-7B-Chat/", "qwen_7b_chat")
QWEN2_5_7B = ("/root/models/Qwen2.5-7B-Instruct/", "qwen2_5_7b")
LLAMA2_7B = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Llama-2-7b-hf/", "llama2_7b")
INTERNLM2_5_7B_CHAT = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/internlm2_5-7b-chat/", "internlm2_5_7b_chat")
YI1_5_6B_CHAT = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Yi-1.5-6B-Chat/", "yi1_5_6b_chat")
QWEN1_5_14B_CHAT = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Qwen1.5-14B-Chat/", "qwen1_5_14b_chat")
LLAMA2_13B_CHAT = ("/root/models/Llama-2-13b-chat-ms/", "llama2_13b_chat")
QWEN_14B_CHAT = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Qwen-14B-Chat/", "qwen_14b_chat")
LLAVA1_5_13B = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/llava-1.5-13b-hf/", "llava1_5_13b")
QWEN1_5_MOE_A2_7B_CHAT = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Qwen1.5-MoE-A2.7B-Chat/", "qwen1_5_moe_a2_7b_chat")
YI1_5_9B_CHAT = ("/root/models/Yi-1.5-9B-Chat/", "yi1_5_9b_chat")
QWEN2_1_5B = ("/mnt/gemininjceph2/geminicephfs/mm-base-plt2/opensource_model/Qwen2-1.5B/", "qwen2_1_5b")
QWEN2_5_72B = ("/root/models/Qwen2.5-72B-Instruct/", "qwen2_5_72b")

WORKLOADS = {
    1: [
        # ('0', DS_R1_DISTILL_QWEN_1_5B), 
        ('1', QWEN2_5_7B),
        ('2', YI1_5_9B_CHAT), 
        ('3', LLAMA2_13B_CHAT),
    ],
    # 2: [
    #     ('0,3', QWEN_7B),
    #     ('1,2', QWEN_7B_CHAT),
    #     ('4,7', LLAMA2_7B),
    #     ('5,6', INTERNLM2_5_7B_CHAT),
    # ],
    # 4: [
    #     # ('0,1,2,3', QWEN2_5_72B),
    #     # ('4,5,6,7', QWEN_7B_CHAT),
    # ],
    # 8: [
    #     ('0,1,2,3,4,5,6,7', QWEN_7B),
    #     ('0,1,2,3,4,5,6,7', QWEN_7B_CHAT),
    # ]
}

def profile_once(
    cuda_visible_devices: str,
    model: str,
    path: str,
    tp: int,
):
    script_path = os.path.join(DIR, '..', 'benchmark', 'benchmark_latency.py')

    cmd = (f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} python3 {script_path} "
       f"--model {model} --enforce-eager "
       f"--dtype float16 --tensor-parallel-size {tp} "
       "--gpu-memory-utilization 0.98")
    
    subpath = f'{DEVICE_TYPE}' if tp == 1 else f'tp{tp}/{DEVICE_TYPE}'
    log_path = os.path.join(DIR, '..', 'logs', f'profile-{path}-{tp}x{DEVICE_TYPE}')
    out_path = os.path.join(DIR, path, subpath)

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    for input_len, batch_size in CASES:
        output_json = os.path.join(out_path, f'i{input_len}b{batch_size}.json')
        output_log = os.path.join(log_path, f'i{input_len}b{batch_size}.log')

        full_cmd = (
            f"{cmd} --input-len {input_len} --batch-size {batch_size} --output-json {output_json} "
            f"2>&1 > {output_log}"
        )

        print(f"------------- Running i{input_len}b{batch_size} -------------")
        print(full_cmd)
        os.system(full_cmd)

if __name__ == '__main__':
    for TP, WORKLOAD in WORKLOADS.items():

        with ProcessPoolExecutor(max_workers=N//TP) as executor:
            it = iter(WORKLOAD)
            while True:
                batch = list(islice(it, N//TP))
                if not batch:
                    break

                print(batch)
                futures = [executor.submit(profile_once, DEVICES, MODEL, PATH, TP) for DEVICES, (MODEL, PATH) in batch]
                for fut in as_completed(futures):
                    pass
