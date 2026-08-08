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

run_dir=${1:?usage: train.sh RUN_DIR}
# shellcheck source=/dev/null
source "$run_dir/control/run.env"

readonly expected_entrypoint="examples/nemo_gym/run_grpo_nemo_gym.py"
readonly expected_config="examples/nemo_gym/grpo_nanov3.yaml"
readonly expected_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
readonly expected_tokenizer="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
tokenizer_id=${TOKENIZER_ID:-$expected_tokenizer}
smoke_mode=${NANOV3_INTERACTIVE_SMOKE:-0}
expected_host_count=${TRAIN_EXPECTED_HOSTS:-32}
readonly expected_gpus_per_host=8
expected_slots_per_host=${TRAIN_SLOTS_PER_HOST:-8}
expected_gpu_total=$((expected_host_count * 8))
readonly success_marker="$run_dir/status/SUCCESS"
readonly failure_marker="$run_dir/status/TRAIN_FAILED"
readonly stop_marker="$run_dir/status/RAY_STOP"
readonly prepared_data_dir="$CACHE_ROOT/huggingface/nanov3_data/prepared"
readonly vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
readonly vllm_venv_dir="$CACHE_ROOT/ray_venvs/$SOURCE_COMMIT/$vllm_actor"
job_id=${LSB_JOBID:-$$}
ray_launch_pids=()

[[ $TARGET_ENTRYPOINT == "$expected_entrypoint" ]] || {
    echo "refusing unexpected entrypoint: ${TARGET_ENTRYPOINT}" >&2
    exit 2
}
[[ $TARGET_CONFIG == "$expected_config" ]] || {
    echo "refusing unexpected config: ${TARGET_CONFIG}" >&2
    exit 2
}
[[ $MODEL_ID == "$expected_model" ]] || {
    echo "refusing unexpected model: ${MODEL_ID}" >&2
    exit 2
}
[[ $tokenizer_id == "$expected_tokenizer" ]] || {
    echo "refusing unexpected tokenizer: ${tokenizer_id}" >&2
    exit 2
}
case "$smoke_mode" in
    0 | 1) ;;
    *)
        echo "NANOV3_INTERACTIVE_SMOKE must be 0 or 1, got: ${smoke_mode}" >&2
        exit 2
        ;;
esac

cleanup() {
    local status=$?
    set +e
    touch "$stop_marker"
    local deadline=$((SECONDS + 90))
    local any_running
    while (( SECONDS < deadline )); do
        any_running=0
        for pid in "${ray_launch_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                any_running=1
                break
            fi
        done
        (( any_running == 0 )) && break
        sleep 3
    done
    for pid in "${ray_launch_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    printf 'finished_at_utc=%s\nexit_status=%s\n' "$(date -u +%FT%TZ)" "$status" \
        >>"$run_dir/logs/train-status.log"
    if (( status != 0 )); then
        printf 'failed_at_utc=%s\nexit_status=%s\njob_id=%s\n' \
            "$(date -u +%FT%TZ)" "$status" "$job_id" >"${failure_marker}.tmp"
        mv "${failure_marker}.tmp" "$failure_marker"
    fi
    exit "$status"
}
trap cleanup EXIT

rm -f "$success_marker" "$failure_marker" "$stop_marker"
find "$run_dir/status/ray" -maxdepth 1 -type f \
    \( -name 'node-*.ready' -o -name 'node-*.failed' \) -delete
mkdir -p \
    "$run_dir/logs/ray" "$run_dir/status/ray" "$run_dir/wandb" \
    "$CHECKPOINT_DIR" "$run_dir/nemo_rl_logs" \
    "$CACHE_ROOT/hf_modules" "$CACHE_ROOT/uv" "$CACHE_ROOT/xdg" \
    "$CACHE_ROOT/inductor" "$CACHE_ROOT/vllm_compile_cache" \
    "$CACHE_ROOT/flashinfer_cubins" "$CACHE_ROOT/flashinfer_workspace" \
    "$CACHE_ROOT/megatron_ckpt_cache" "$CACHE_ROOT/hf_config_locks"

printf 'started_at_utc=%s\njob_id=%s\nhost=%s\n' \
    "$(date -u +%FT%TZ)" "$job_id" "$(hostname -f)" \
    >"$run_dir/logs/train-status.log"

[[ -s "$run_dir/status/PREP_SUCCESS" ]] || {
    echo "PREP_SUCCESS is missing: ${run_dir}/status/PREP_SUCCESS" >&2
    exit 2
}
[[ $(git -C "$run_dir/source" rev-parse HEAD) == "$SOURCE_COMMIT" ]] || {
    echo "source checkout changed after preparation" >&2
    exit 2
}
[[ -s $CONTAINER_IMAGE ]] || {
    echo "container image is missing: ${CONTAINER_IMAGE}" >&2
    exit 2
}
[[ -s "$prepared_data_dir/train-split.jsonl" ]]
[[ -s "$prepared_data_dir/val-split.jsonl" ]]
[[ -s "$CACHE_ROOT/huggingface/gym_venvs/GYM_SUCCESS" ]]
[[ -s "$vllm_venv_dir/VENV_SUCCESS" && -L "$vllm_venv_dir/bin/python" ]]
[[ -x "$run_dir/control/ray_node.sh" ]]
command -v blaunch >/dev/null || {
    echo "blaunch is unavailable" >&2
    exit 2
}

[[ -n ${LSB_MCPU_HOSTS:-} ]] || {
    echo "LSB_MCPU_HOSTS is unset; this payload must run in an LSF allocation" >&2
    exit 2
}
read -r -a host_slot_fields <<<"$LSB_MCPU_HOSTS"
(( ${#host_slot_fields[@]} % 2 == 0 )) || {
    echo "malformed LSB_MCPU_HOSTS: ${LSB_MCPU_HOSTS}" >&2
    exit 2
}

hosts=()
for ((i = 0; i < ${#host_slot_fields[@]}; i += 2)); do
    host=${host_slot_fields[i]}
    slots=${host_slot_fields[i + 1]}
    [[ $slots == "$expected_slots_per_host" ]] || {
        echo "expected ${expected_slots_per_host} LSF slots on ${host}, found ${slots}" >&2
        exit 2
    }
    hosts+=("$host")
done
[[ ${#hosts[@]} -eq $expected_host_count ]] || {
    echo "expected ${expected_host_count} allocated hosts, found ${#hosts[@]}: ${hosts[*]}" >&2
    exit 2
}

head_host=${hosts[0]}
current_host=$(hostname -s)
[[ ${head_host%%.*} == "$current_host" ]] || {
    echo "LSF driver host ${current_host} is not first allocated host ${head_host}" >&2
    exit 2
}
if [[ ${head_host%%.*} == "$current_host" ]]; then
    # Ray's control plane is TCP, but bind it to the Blue Vela IPoIB address;
    # NCCL collectives remain on InfiniBand verbs via NCCL_IB_HCA.
    head_ip=$(ip -4 -o addr show dev ibp26s0 scope global 2>/dev/null \
        | awk '{sub(/\/.*$/, "", $4); print $4; exit}')
    head_ip=${head_ip:-$(hostname -I 2>/dev/null | tr ' ' '\n' \
        | awk '/^[0-9]+(\.[0-9]+){3}$/ && $1 !~ /^127\./ {print; exit}')}
else
    head_ip=$(getent ahostsv4 "$head_host" | awk '$3 == "STREAM" {print $1; exit}')
fi
[[ -n $head_ip ]] || {
    echo "unable to resolve IPv4 address for Ray head ${head_host}" >&2
    exit 2
}
head_address="${head_ip}:1200"

printf '%s\n' "${hosts[@]}" >"$run_dir/status/allocated-hosts.txt"
printf 'head_host=%s\nhead_ip=%s\nhead_address=%s\n' \
    "$head_host" "$head_ip" "$head_address" >"$run_dir/status/ray-topology.txt"
echo "Validated LSF allocation: ${#hosts[@]} hosts, ${expected_slots_per_host} slots each"
echo "Ray head: ${head_host} (${head_address})"

launch_ray_node() {
    local host=$1
    local role=$2
    local rank=$3
    blaunch -z "$host" /bin/bash "$run_dir/control/ray_node.sh" \
        "$run_dir" "$role" "$head_address" "$rank" \
        >"$run_dir/logs/ray/node-${rank}-${host}.log" 2>&1 &
    ray_launch_pids+=("$!")
}

wait_for_node_marker() {
    local rank=$1
    local timeout_seconds=$2
    local deadline=$((SECONDS + timeout_seconds))
    while [[ ! -s "$run_dir/status/ray/node-${rank}.ready" ]]; do
        if [[ -s "$run_dir/status/ray/node-${rank}.failed" ]]; then
            echo "Ray node ${rank} failed; see logs/ray" >&2
            return 1
        fi
        (( SECONDS < deadline )) || {
            echo "timed out waiting for Ray node ${rank}" >&2
            return 1
        }
        sleep 2
    done
}

launch_ray_node "$head_host" head 0
wait_for_node_marker 0 300

for ((rank = 1; rank < expected_host_count; rank++)); do
    launch_ray_node "${hosts[rank]}" worker "$rank"
done
for ((rank = 1; rank < expected_host_count; rank++)); do
    wait_for_node_marker "$rank" 1800
done
echo "All ${expected_host_count} Ray node processes reported ready"

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

head_node_tmp="/tmp/nrl-nanov3-ray-${job_id}-0"
case "$head_node_tmp" in
    /tmp/nrl-nanov3-ray-[0-9]*-0) ;;
    *)
        echo "refusing unsafe head-node temp path: ${head_node_tmp}" >&2
        exit 2
        ;;
esac
[[ -d $head_node_tmp ]] || {
    echo "Ray head temporary directory is missing: ${head_node_tmp}" >&2
    exit 2
}

container=(
    "$APPTAINER" exec
    --fakeroot
    --nv
    --contain
    --writable-tmpfs
    --no-mount home
    --bind "$run_dir/source:/opt/nemo-rl"
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$head_node_tmp:/job-tmp"
    --bind "$CACHE_ROOT/huggingface/gym_venvs:/opt/gym_venvs"
    --bind "$vllm_venv_dir:/opt/ray_venvs/$vllm_actor"
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
    --env "RAY_ADDRESS=$head_address"
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

"${container[@]}" python - "$head_address" "$head_ip" "$expected_host_count" "$expected_gpu_total" <<'PY'
import sys
import time

import ray

address, node_ip, expected_nodes, expected_gpus = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
)
ray.init(address=address, _node_ip_address=node_ip, log_to_driver=True)
deadline = time.monotonic() + 900
while True:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    gpus = int(ray.cluster_resources().get("GPU", 0))
    if len(nodes) == expected_nodes and gpus == expected_gpus:
        print(f"Ray cluster ready: nodes={len(nodes)} GPUs={gpus}", flush=True)
        break
    if time.monotonic() >= deadline:
        raise RuntimeError(
            f"Ray cluster did not reach nodes={expected_nodes}, GPUs={expected_gpus}; "
            f"observed nodes={len(nodes)}, GPUs={gpus}"
        )
    print(
        f"Waiting for Ray resources: nodes={len(nodes)}/{expected_nodes}, "
        f"GPUs={gpus}/{expected_gpus}",
        flush=True,
    )
    time.sleep(5)
ray.shutdown()
PY

driver_log="$run_dir/logs/train-driver.log"
set +e
# Positional parameters are expanded by the container-side shell.
# shellcheck disable=SC2016
"${container[@]}" /bin/bash -ce '
model_id=$1
tokenizer_id=$2
data_dir=$3
run_dir=$4
run_id=$5
checkpoint_dir=$6
expected_host_count=$7
smoke_mode=$8
smoke_overrides=()
if [[ $smoke_mode == 1 ]]; then
    echo "Interactive smoke overrides enabled"
    smoke_overrides=(
        grpo.num_prompts_per_step=2
        grpo.num_generations_per_prompt=4
        grpo.max_num_steps=1
        policy.train_global_batch_size=4
        policy.generation.max_new_tokens=512
        checkpointing.enabled=false
    )
fi
cd /opt/nemo-rl
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/nemo_gym/grpo_nanov3.yaml \
    "cluster.num_nodes=$expected_host_count" \
    "policy.model_name=$model_id" \
    "policy.tokenizer.name=$tokenizer_id" \
    "+policy.generation.vllm_kwargs.tokenizer=$tokenizer_id" \
    "data.train.data_path=$data_dir/train-split.jsonl" \
    "data.validation.data_path=$data_dir/val-split.jsonl" \
    "logger.log_dir=$run_dir/nemo_rl_logs" \
    logger.wandb_enabled=true \
    logger.wandb.project=nemo-rl \
    "logger.wandb.name=$run_id" \
    logger.tensorboard_enabled=true \
    "checkpointing.checkpoint_dir=$checkpoint_dir" \
    "${smoke_overrides[@]}"
' -- "$MODEL_ID" "$tokenizer_id" "$prepared_data_dir" "$run_dir" "$RUN_ID" "$CHECKPOINT_DIR" \
    "$expected_host_count" "$smoke_mode" \
    2>&1 | tee "$driver_log"
driver_status=${PIPESTATUS[0]}
set -e
printf 'driver_exit_status=%s\n' "$driver_status" >>"$run_dir/logs/train-status.log"
(( driver_status == 0 )) || exit "$driver_status"

wandb_url=$(grep -Eo 'https://wandb\.ai/[^[:space:]]+' "$driver_log" | tail -n 1 || true)
{
    printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'job_id=%s\n' "$job_id"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'entrypoint=%s\n' "$TARGET_ENTRYPOINT"
    printf 'config=%s\n' "$TARGET_CONFIG"
    printf 'model=%s\n' "$MODEL_ID"
    printf 'hosts=%s\n' "$expected_host_count"
    printf 'gpus=%s\n' "$expected_gpu_total"
    printf 'wandb_url=%s\n' "${wandb_url:-not-found-in-console-log}"
} >"${success_marker}.tmp"
mv "${success_marker}.tmp" "$success_marker"

echo "Training complete: ${success_marker}"
