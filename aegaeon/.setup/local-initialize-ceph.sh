#! /bin/bash
set -ex

if [ -z "$PREFIX" ]; then
    echo "PREFIX is not set, please pass it to jump-initialize-ceph."
    exit 1
fi

if [ -z "$MNT_POINT" ]; then
    echo "MNT_POINT is not set, please pass it to jump-initialize-ceph."
    exit 1
fi

THIS_DIR=`realpath $(dirname $0)`
ROOT_DIR=`realpath $THIS_DIR/..`
REPO_NAME=$(basename $ROOT_DIR)

# echo "export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
# echo "export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
# echo "export http_proxy=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
# echo "export https_proxy=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
# echo "export NO_PROXY=127.0.0.1,0.0.0.0" >> ~/.bashrc
echo "alias proxy_on='export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113; export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113; export http_proxy=http://hk-mmhttpproxy.woa.com:11113; export https_proxy=http://hk-mmhttpproxy.woa.com:11113;'" >> ~/.bashrc
echo "alias proxy_off='unset HTTP_PROXY; unset HTTPS_PROXY; unset http_proxy; unset https_proxy'" >> ~/.bashrc


function proxy_on {
    export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113
    export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113
    export http_proxy=http://hk-mmhttpproxy.woa.com:11113
    export https_proxy=http://hk-mmhttpproxy.woa.com:11113
    export NO_PROXY=127.0.0.1,0.0.0.0
}

function proxy_off {
    unset HTTP_PROXY
    unset HTTPS_PROXY
    unset http_proxy
    unset https_proxy
    unset NO_PROXY
}

proxy_on

if [ -z "$SKIP_APT" ]; then
    apt update
    apt install -y net-tools tmux iputils-ping htop nvtop
fi

CONDA="/root/miniconda3/bin/conda"
CONDA_ENV=$PREFIX

if ! $CONDA env list | grep -q "^$CONDA_ENV\s"; then
    $CONDA create --name "$CONDA_ENV" python=3.10.12 --yes
    echo "Created new conda env: $CONDA_ENV"
else
    echo "Conda env '$CONDA_ENV' already exists. Skipping."
fi

# Disable auto-activation of base environment
$CONDA config --set auto_activate_base false

proxy_off

CONDA_ENV_BIN=/root/miniconda3/envs/$CONDA_ENV/bin
PYTHON_COMMAND="$CONDA_ENV_BIN/python3.10"
PIP_COMMAND="$PYTHON_COMMAND -m pip"


$PIP_COMMAND config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
$PIP_COMMAND config set global.trusted-host mirrors.cloud.tencent.com


# Add NCCL and networking environment variables idempotently
function add_env_var_if_not_exists() {
    local var_name="$1"
    local var_value="$2"
    local export_line="export $var_name=$var_value"

    if ! grep -q "^export $var_name=" ~/.bashrc; then
        echo "$export_line" >> ~/.bashrc
    fi
}

add_env_var_if_not_exists "NCCL_IB_GID_INDEX" "3"
add_env_var_if_not_exists "NCCL_IB_SL" "3"
add_env_var_if_not_exists "NCCL_CHECK_DISABLE" "1"
add_env_var_if_not_exists "NCCL_P2P_DISABLE" "0"
add_env_var_if_not_exists "NCCL_IB_DISABLE" "0"
add_env_var_if_not_exists "NCCL_LL_THRESHOLD" "16384"
add_env_var_if_not_exists "NCCL_IB_CUDA_SUPPORT" "1"
add_env_var_if_not_exists "NCCL_SOCKET_IFNAME" "bond1"
add_env_var_if_not_exists "UCX_NET_DEVICES" "bond1"
add_env_var_if_not_exists "NCCL_IB_HCA" "mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6"
add_env_var_if_not_exists "NCCL_COLLNET_ENABLE" "0"
add_env_var_if_not_exists "SHARP_COLL_ENABLE_SAT" "0"
add_env_var_if_not_exists "NCCL_NET_GDR_LEVEL" "2"
add_env_var_if_not_exists "NCCL_IB_QPS_PER_CONNECTION" "4"
add_env_var_if_not_exists "NCCL_IB_TC" "160"
add_env_var_if_not_exists "NCCL_PXN_DISABLE" "0"

# Add ulimit setting if not exists
if ! grep -q "ulimit -n 65536" ~/.bashrc; then
    echo "ulimit -n 65536" >> ~/.bashrc
fi

source ~/.bashrc


# Install NIXL

proxy_on

cd $MNT_POINT

# if nixl does not exist, return error
if [ ! -d "nixl" ]; then
    echo "nixl does not exist, please git clone in your ceph mount point first."
    exit 1
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd nixl
$PIP_COMMAND install meson --trusted-host mirrors.cloud.tencent.com
$PIP_COMMAND uninstall -y nixl || true
$PIP_COMMAND install . --config-settings=setup-args="-Ducx_path=/usr/local/ucx" --trusted-host mirrors.cloud.tencent.com
# loading nixl's .so files requires `GLIBCXX_3.4.30`, not available in conda by default
$CONDA install -n $PREFIX -c conda-forge -y gcc=12.1.0

proxy_off

# if REINSTALL_NIXL_ONLY is set, return
if [ -n "$REINSTALL_NIXL_ONLY" ]; then
    echo "REINSTALL_NIXL_ONLY is set, exiting."
    exit 0
fi

# 压测工具
$PIP_COMMAND uninstall -y libra-openai-benchmark
# This may fail. But we only need it on node0
# Dirty hack to let it pass
$PIP_COMMAND install -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com libra-openai-benchmark || true

# 下载luban工具
$PIP_COMMAND install lubanml --index-url=https://mirrors.tencent.com/pypi/simple --extra-index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

# 拉取模型
# 点击右上角头像获取token：https://lubanml.woa.com/#/
# $PYTHON_COMMAND -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_r1:model_path'); print(ret)"
# $PYTHON_COMMAND -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_v3:model_path'); print(ret)"
$PYTHON_COMMAND -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_r1_distill_qwen_1_5b:model_path'); print(ret)"


$PIP_COMMAND uninstall -y sglang
SGLANG_LOCALTION=$MNT_POINT/$REPO_NAME

cd $SGLANG_LOCALTION
$PIP_COMMAND install -e "python[all]" --trusted-host mirrors.cloud.tencent.com
$PIP_COMMAND install "nvidia-nccl-cu12==2.25.1" --no-deps --trusted-host mirrors.cloud.tencent.com

# $PIP_COMMAND install sgl-kernel --force-reinstall
# $PIP_COMMAND uninstall -y flashinfer-python && $PIP_COMMAND install flashinfer-python #  -i https://flashinfer.ai/whl/cu124/torch2.5


# 减少TIME_WAIT状态的持续时间（默认60秒）
sysctl -w net.ipv4.tcp_fin_timeout=30
# 启用TIME_WAIT端口重用
sysctl -w net.ipv4.tcp_tw_reuse=1
# 增加本地端口范围
sysctl -w net.ipv4.ip_local_port_range="1024 65535"
# 使更改永久生效
sysctl -p

echo "All done"
