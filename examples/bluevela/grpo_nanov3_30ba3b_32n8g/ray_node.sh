#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Keep these values inside the container command.  Apptainer's --env parser
# treats commas as separators, while Blue Vela's NCCL allowlists contain them.
export NCCL_IB_HCA='^=mlx5_1,mlx5_6'
export NCCL_SOCKET_IFNAME='=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0'

run_dir=${1:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
role=${2:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
head_address=${3:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
rank=${4:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
# shellcheck source=/dev/null
source "$run_dir/control/run.env"
expected_host_count=${RAY_EXPECTED_HOSTS:-32}

[[ $role == head || $role == worker ]] || {
    echo "unsupported Ray role: ${role}" >&2
    exit 2
}
[[ $rank =~ ^[0-9]+$ && $rank -lt $expected_host_count ]] || {
    echo "invalid Ray node rank: ${rank}" >&2
    exit 2
}
[[ $head_address =~ ^[^:]+:[0-9]+$ ]] || {
    echo "invalid Ray head address: ${head_address}" >&2
    exit 2
}

job_id=${LSB_JOBID:-$$}
node_tmp="/tmp/nrl-nanov3-ray-${job_id}-${rank}"
ready_marker="$run_dir/status/ray/node-${rank}.ready"
failure_marker="$run_dir/status/ray/node-${rank}.failed"
stop_marker="$run_dir/status/RAY_STOP"
vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
vllm_venv_dir="$CACHE_ROOT/ray_venvs/$SOURCE_COMMIT/$vllm_actor"

case "$node_tmp" in
    /tmp/nrl-nanov3-ray-[0-9]*-[0-9]*) ;;
    *)
        echo "refusing unsafe temporary path: ${node_tmp}" >&2
        exit 2
        ;;
esac

cleanup() {
    local status=$?
    if (( status != 0 )); then
        printf 'failed_at_utc=%s\nexit_status=%s\nhost=%s\nrole=%s\nrank=%s\n' \
            "$(date -u +%FT%TZ)" "$status" "$(hostname -f)" "$role" "$rank" \
            >"${failure_marker}.tmp"
        mv "${failure_marker}.tmp" "$failure_marker"
    fi
    rm -f "$ready_marker"
    rm -rf -- "$node_tmp"
    exit "$status"
}
trap cleanup EXIT

rm -f "$ready_marker" "$failure_marker"
mkdir -p \
    "$node_tmp/ray" "$node_tmp/torch" "$node_tmp/triton" \
    "$run_dir/status/ray" "$CACHE_ROOT/hf_modules" "$CACHE_ROOT/uv" \
    "$CACHE_ROOT/xdg" "$CACHE_ROOT/inductor" \
    "$CACHE_ROOT/vllm_compile_cache" "$CACHE_ROOT/flashinfer_cubins" \
    "$CACHE_ROOT/flashinfer_workspace" "$CACHE_ROOT/megatron_ckpt_cache" \
    "$CACHE_ROOT/hf_config_locks"

[[ -s $CONTAINER_IMAGE ]] || {
    echo "container image is missing: ${CONTAINER_IMAGE}" >&2
    exit 2
}
[[ -s "$run_dir/status/PREP_SUCCESS" ]] || {
    echo "PREP_SUCCESS is missing" >&2
    exit 2
}
[[ -s "$vllm_venv_dir/VENV_SUCCESS" && -L "$vllm_venv_dir/bin/python" ]] || {
    echo "vLLM actor environment is incomplete: ${vllm_venv_dir}" >&2
    exit 2
}

command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable on $(hostname -f)" >&2
    exit 2
}
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#gpu_names[@]} -eq 8 ]] || {
    echo "expected 8 visible GPUs on $(hostname -f), found ${#gpu_names[@]}" >&2
    printf '%s\n' "${gpu_names[@]}" >&2
    exit 2
}
for gpu_name in "${gpu_names[@]}"; do
    [[ $gpu_name == *H100* ]] || {
        echo "expected H100 on $(hostname -f), found: ${gpu_name}" >&2
        exit 2
    }
done

read_secret() {
    local name=$1
    local path=$2
    local mode value
    [[ -f $path && ! -L $path && -s $path ]] || {
        echo "${name} file is missing or unsafe: ${path}" >&2
        return 1
    }
    mode=$(stat -c %a "$path")
    (( (8#$mode & 077) == 0 )) || {
        echo "${name} file permissions are too broad: ${path}" >&2
        return 1
    }
    IFS= read -r value <"$path" || true
    value=${value%$'\r'}
    [[ -n $value ]] || {
        echo "${name} file is empty: ${path}" >&2
        return 1
    }
    printf -v "$name" '%s' "$value"
    export "${name?}"
}

read_secret HF_TOKEN "$HF_TOKEN_FILE"
read_secret WANDB_API_KEY "$WANDB_API_KEY_FILE"
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"
export APPTAINERENV_NCCL_IB_HCA='^=mlx5_1,mlx5_6'
export APPTAINERENV_NCCL_SOCKET_IFNAME='=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0'
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_DEBUG_SUBSYS=INIT,NET

shopt -s nullglob
rdma_devices=(/dev/infiniband/*)
shopt -u nullglob
(( ${#rdma_devices[@]} > 0 )) || {
    echo "Host RDMA devices are missing on $(hostname -s)" >&2
    exit 2
}
printf 'Host RDMA devices on %s:' "$(hostname -s)"
printf ' %s' "${rdma_devices[@]##*/}"
printf '\n'

container=(
    "$APPTAINER" exec
    --fakeroot
    --nv
    --containall
    --writable-tmpfs
    --no-mount home
    --bind "$run_dir/source:/opt/nemo-rl"
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$node_tmp:/job-tmp"
    --bind "$CACHE_ROOT/huggingface/gym_venvs:/opt/gym_venvs"
    --bind "$vllm_venv_dir:/opt/ray_venvs/$vllm_actor"
    --bind /dev/infiniband:/dev/infiniband
    --bind /etc/resolv.conf:/etc/resolv.conf:ro
    --pwd /opt/nemo-rl
    --env "USER=$(id -un)"
    --env "HF_HOME=$CACHE_ROOT/huggingface"
    --env "HF_MODULES_CACHE=$CACHE_ROOT/hf_modules"
    --env "XDG_CACHE_HOME=$CACHE_ROOT/xdg"
    --env "UV_CACHE_DIR=$CACHE_ROOT/uv"
    --env UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    --env VIRTUAL_ENV=/opt/nemo_rl_venv
    --env UV_NO_SYNC=1
    --env UV_LINK_MODE=copy
    --env UV_HTTP_TIMEOUT=600
    --env NRL_CONTAINER=1
    --env NRL_IGNORE_VERSION_MISMATCH=1
    --env NRL_WG_USE_RAY_REF=1
    --env NRL_VLLM_USE_V1=1
    --env NEMO_RL_VENV_DIR=/opt/ray_venvs
    --env NEMO_GYM_VENV_DIR=/opt/gym_venvs
    --env "PERSISTENT_CACHE=$CACHE_ROOT"
    --env RAY_TMPDIR=/job-tmp/ray
    --env TMPDIR=/job-tmp
    --env TORCH_HOME=/job-tmp/torch
    --env TRITON_CACHE_DIR=/job-tmp/triton
    --env "TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor"
    --env "VLLM_CACHE_ROOT=$CACHE_ROOT/vllm_compile_cache"
    --env "DG_JIT_CACHE_DIR=$CACHE_ROOT/vllm_compile_cache/deep_gemm"
    --env "FLASHINFER_CUBIN_DIR=$CACHE_ROOT/flashinfer_cubins"
    --env "FLASHINFER_CUBIN_CACHE=$CACHE_ROOT/flashinfer_cubins"
    --env "FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer_workspace"
    --env "WANDB_DIR=$run_dir/wandb"
    --env "WANDB_CACHE_DIR=$CACHE_ROOT/wandb"
    --env "NRL_MEGATRON_CHECKPOINT_DIR=$CACHE_ROOT/megatron_ckpt_cache"
    --env "MEGATRON_CONFIG_LOCK_DIR=$CACHE_ROOT/hf_config_locks"
    --env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
    --env TOKENIZERS_PARALLELISM=false
    --env VLLM_DEEP_GEMM_WARMUP=skip
    --env NCCL_IB_QPS_PER_CONNECTION=2
    --env NCCL_IB_SPLIT_DATA_ON_QPS=0
    --env NCCL_IB_TIMEOUT=16
    --env NCCL_IB_RETRY_CNT=14
    --env NCCL_IB_DISABLE=0
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    --env TORCH_CUDA_ARCH_LIST=9.0
    --env UCX_NET_DEVICES=mlx5_0:1
    --env PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    "$CONTAINER_IMAGE"
)
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    container=("${container[@]:0:${#container[@]}-1}" \
        --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$CONTAINER_IMAGE")
fi

container_start_attempts=${CONTAINER_START_ATTEMPTS:-3}
[[ $container_start_attempts =~ ^[1-9][0-9]*$ ]] || {
    echo "CONTAINER_START_ATTEMPTS must be a positive integer" >&2
    exit 2
}

run_container_with_startup_retry() {
    local attempt status
    for ((attempt = 1; attempt <= container_start_attempts; attempt++)); do
        if "${container[@]}" "$@"; then
            return 0
        else
            status=$?
        fi
        if (( status != 255 || attempt == container_start_attempts )); then
            return "$status"
        fi
        echo "Apptainer startup returned status 255 on Ray node rank ${rank};" \
            " retrying in 10 seconds (${attempt}/${container_start_attempts})." >&2
        sleep 10
    done
}

run_container_with_startup_retry /bin/bash -ce '
shopt -s nullglob
rdma_devices=(/dev/infiniband/*)
rdma_hcas=(/sys/class/infiniband/*)
shopt -u nullglob
(( ${#rdma_devices[@]} > 0 )) || {
    echo "Container RDMA devices are missing" >&2
    exit 2
}
(( ${#rdma_hcas[@]} > 0 )) || {
    echo "Container RDMA HCAs are missing" >&2
    exit 2
}
printf "Container RDMA devices:"
printf " %s" "${rdma_devices[@]##*/}"
printf "\nContainer RDMA HCAs:"
printf " %s" "${rdma_hcas[@]##*/}"
printf "\n"
'

read -r -d '' ray_node_inner <<'INNER' || true
set -euo pipefail

role=$1
head_address=$2
rank=$3
ready_marker=$4
stop_marker=$5
head_ip=${head_address%:*}
topo_rank=$((rank + 1))

ray stop --force >/dev/null 2>&1 || true

common_args=(
    --disable-usage-stats
    --num-cpus=17
    --num-gpus=8
    --resources "{\"topo_rank\": ${topo_rank}}"
    --min-worker-port=2000
    --max-worker-port=2999
)

if [[ $role == head ]]; then
    ray start --head \
        --node-ip-address="$head_ip" \
        --port=1200 \
        --ray-client-server-port=1201 \
        --dashboard-port=8265 \
        --dashboard-host="$head_ip" \
        --include-dashboard=true \
        --node-manager-port=1302 \
        --object-manager-port=1304 \
        --runtime-env-agent-port=1306 \
        --dashboard-agent-grpc-port=1308 \
        --dashboard-agent-listen-port=1312 \
        --metrics-export-port=1310 \
        "${common_args[@]}"
else
    started=0
    for attempt in $(seq 1 30); do
        if ray start --address="$head_address" \
            --node-manager-port=1301 \
            --object-manager-port=1303 \
            --runtime-env-agent-port=1305 \
            --dashboard-agent-grpc-port=1307 \
            --dashboard-agent-listen-port=1311 \
            --metrics-export-port=1309 \
            "${common_args[@]}"; then
            started=1
            break
        fi
        echo "Ray worker rank ${rank} start attempt ${attempt}/30 failed"
        ray stop --force >/dev/null 2>&1 || true
        sleep 10
    done
    (( started == 1 )) || {
        echo "Ray worker rank ${rank} failed to connect to ${head_address}" >&2
        exit 1
    }
fi

printf 'ready_at_utc=%s\nhost=%s\nrole=%s\nrank=%s\nhead_address=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname -f)" "$role" "$rank" "$head_address" \
    >"${ready_marker}.tmp"
mv "${ready_marker}.tmp" "$ready_marker"

while [[ ! -f $stop_marker ]]; do
    sleep 10
    ray status --address="$head_address" >/dev/null
done

ray stop --force
INNER

run_container_with_startup_retry /bin/bash -ce "$ray_node_inner" -- \
    "$role" "$head_address" "$rank" "$ready_marker" "$stop_marker"
