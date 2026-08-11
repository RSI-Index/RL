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

run_dir=${1:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
role=${2:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
head_address=${3:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
rank=${4:?usage: ray_node.sh RUN_DIR ROLE HEAD_ADDRESS RANK}
source "$run_dir/control/run.env"

readonly expected_host_count=27
readonly vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
readonly vllm_archive_tag="vllm-0.25.1-nccl-2.28.9"
readonly vllm_archive="$ARCHIVE_ROOT/ray_venvs/$SOURCE_COMMIT/${vllm_actor}.${vllm_archive_tag}.tar"
readonly vllm_archive_marker="${vllm_archive}.success"
readonly gym_archive="$ARCHIVE_ROOT/huggingface/gym_venvs.tar"
readonly gym_archive_marker="${gym_archive}.success"
job_id=${LSB_JOBID:-$$}
node_tmp="/tmp/nrl-super-27n-ray-${job_id}-${rank}"
ready_marker="$run_dir/status/ray/node-${rank}.ready"
failure_marker="$run_dir/status/ray/node-${rank}.failed"
stop_marker="$run_dir/status/RAY_STOP"

[[ $role == head || $role == worker ]] || { echo "unsupported Ray role" >&2; exit 2; }
[[ $rank =~ ^[0-9]+$ && $rank -lt $expected_host_count ]] || { echo "invalid rank" >&2; exit 2; }
[[ $head_address =~ ^[^:]+:[0-9]+$ ]] || { echo "invalid head address" >&2; exit 2; }

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
mkdir -p "$node_tmp/ray" "$node_tmp/torch" "$node_tmp/triton" \
    "$node_tmp/gym_venvs" "$node_tmp/vllm_worker" "$node_tmp/hf_modules" \
    "$node_tmp/uv" "$node_tmp/xdg" "$node_tmp/inductor" \
    "$node_tmp/vllm_compile_cache/deep_gemm" "$node_tmp/flashinfer_cubins" \
    "$node_tmp/flashinfer_workspace" "$run_dir/status/ray" \
    "$CACHE_ROOT/megatron_ckpt_cache" "$CACHE_ROOT/hf_config_locks"
mkdir -p "$JUDGE_HF_HOME/xet"

[[ -s $CONTAINER_IMAGE ]] || { echo "container image is missing" >&2; exit 2; }
[[ -s $vllm_archive && -s $vllm_archive_marker ]] || {
    echo "vLLM worker archive is incomplete" >&2
    exit 2
}
[[ -s $gym_archive && -s $gym_archive_marker ]] || { echo "Gym archive is incomplete" >&2; exit 2; }
tar -C "$node_tmp/vllm_worker" -xf "$vllm_archive"
tar -C "$node_tmp/gym_venvs" -xf "$gym_archive"
[[ -L $node_tmp/vllm_worker/bin/python && -s $node_tmp/vllm_worker/VENV_SUCCESS ]] || {
    echo "node-local vLLM worker environment is incomplete" >&2
    exit 2
}
find "$node_tmp/gym_venvs" -mindepth 1 -maxdepth 3 -type d -name .venv -print -quit | grep -q . || {
    echo "node-local Gym environments are incomplete" >&2
    exit 2
}
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 2; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#gpu_names[@]} -eq 8 ]] || { echo "expected 8 visible GPUs" >&2; exit 2; }
for gpu_name in "${gpu_names[@]}"; do
    [[ $gpu_name == *H100* ]] || { echo "expected H100, found $gpu_name" >&2; exit 2; }
done

head_ip=${head_address%:*}
if [[ $role == head ]]; then
    node_ip=$head_ip
else
    read -r node_interface node_ip < <(
        ip -4 route get "$head_ip" | awk '
            { for (i = 1; i <= NF; i++) {
                if ($i == "dev") dev = $(i + 1)
                if ($i == "src") src = $(i + 1)
            } }
            END { print dev, src }
        '
    )
fi
node_interface=${node_interface:-$(
    ip -4 -o addr show scope global | awk -v expected_ip="$node_ip" '
        { ip = $4; sub(/\/.*/, "", ip) }
        ip == expected_ip { sub(/@.*/, "", $2); print $2; exit }
    '
)}
[[ -n $node_ip && -n $node_interface ]] || {
    echo "unable to resolve the Ray/NCCL network path to $head_ip" >&2
    exit 2
}
echo "Ray/NCCL network: node_ip=$node_ip interface=$node_interface head_ip=$head_ip"

export APPTAINERENV_NCCL_IB_HCA='^=mlx5_1,mlx5_6'
export APPTAINERENV_NCCL_SOCKET_IFNAME="=${node_interface}"
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_DEBUG_SUBSYS=INIT,NET

shopt -s nullglob
rdma_devices=(/dev/infiniband/*)
shopt -u nullglob
(( ${#rdma_devices[@]} > 0 )) || { echo "Host RDMA devices are missing" >&2; exit 2; }

container=(
    "$APPTAINER" exec --fakeroot --nv --contain --writable-tmpfs --no-mount home
    --bind "$SOURCE_DIR:/opt/nemo-rl"
    --bind /proj:/proj
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$node_tmp:/job-tmp"
    --bind "$node_tmp/gym_venvs:/opt/gym_venvs"
    --bind "$node_tmp/vllm_worker:/opt/ray_venvs/$vllm_actor"
    --bind /dev/infiniband:/dev/infiniband
    --bind /etc/resolv.conf:/etc/resolv.conf:ro
    --pwd /opt/nemo-rl
    --env "HF_HOME=$JUDGE_HF_HOME"
    --env "HF_XET_CACHE=$JUDGE_HF_HOME/xet"
    --env HF_MODULES_CACHE=/job-tmp/hf_modules
    --env XDG_CACHE_HOME=/job-tmp/xdg
    --env UV_CACHE_DIR=/job-tmp/uv
    --env UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    --env VIRTUAL_ENV=/opt/nemo_rl_venv
    --env UV_NO_SYNC=1
    --env UV_LINK_MODE=copy
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
    --env TORCHINDUCTOR_CACHE_DIR=/job-tmp/inductor
    --env VLLM_CACHE_ROOT=/job-tmp/vllm_compile_cache
    --env DG_JIT_CACHE_DIR=/job-tmp/vllm_compile_cache/deep_gemm
    --env FLASHINFER_CUBIN_DIR=/job-tmp/flashinfer_cubins
    --env FLASHINFER_WORKSPACE_BASE=/job-tmp/flashinfer_workspace
    --env RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
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

"${container[@]}" /bin/bash -ce '
set -euo pipefail
ray stop --force >/dev/null 2>&1 || true
common_args=(
  --disable-usage-stats
  --num-cpus=17
  --num-gpus=8
  --resources "{\"topo_rank\": $(( $3 + 1 ))}"
  --min-worker-port=2000
  --max-worker-port=2999
)
if [[ $1 == head ]]; then
  ray start --head --node-ip-address="${2%:*}" --port=1200 \
    --ray-client-server-port=1201 --dashboard-port=8265 --dashboard-host="${2%:*}" \
    --include-dashboard=true --node-manager-port=1302 --object-manager-port=1304 \
    --runtime-env-agent-port=1306 --dashboard-agent-grpc-port=1308 \
    --dashboard-agent-listen-port=1312 --metrics-export-port=1310 "${common_args[@]}"
else
  started=0
  for attempt in $(seq 1 30); do
    if ray start --address="$2" --node-ip-address="$6" \
      --node-manager-port=1301 --object-manager-port=1303 \
      --dashboard-agent-grpc-port=1307 --dashboard-agent-listen-port=1311 \
      --metrics-export-port=1309 "${common_args[@]}"; then
      started=1
      break
    fi
    ray stop --force >/dev/null 2>&1 || true
    sleep 10
  done
  (( started == 1 )) || exit 1
fi
printf "ready_at_utc=%s\nhost=%s\nrole=%s\nrank=%s\nhead_address=%s\n" \
  "$(date -u +%FT%TZ)" "$(hostname -f)" "$1" "$3" "$2" >"$4.tmp"
mv "$4.tmp" "$4"
while [[ ! -f $5 ]]; do
  sleep 10
  ray status --address="$2" >/dev/null
done
if [[ -d /job-tmp/ray/session_latest/logs ]]; then
  tar -C /job-tmp/ray/session_latest -czf \
    "$7/logs/ray/raw-node-$3-$(hostname -s).tar.gz.tmp" logs
  mv "$7/logs/ray/raw-node-$3-$(hostname -s).tar.gz.tmp" \
    "$7/logs/ray/raw-node-$3-$(hostname -s).tar.gz"
fi
ray stop --force
' -- "$role" "$head_address" "$rank" "$ready_marker" "$stop_marker" "$node_ip" "$run_dir"
