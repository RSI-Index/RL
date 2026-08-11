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
source "$run_dir/control/run.env"

readonly expected_entrypoint="examples/nemo_gym/run_grpo_nemo_gym.py"
readonly expected_config="examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml"
readonly expected_host_count=27
readonly expected_gpus_per_host=8
readonly expected_slots_per_host=17
readonly expected_gpu_total=$((expected_host_count * expected_gpus_per_host))
readonly success_marker="$run_dir/status/SUCCESS"
readonly failure_marker="$run_dir/status/TRAIN_FAILED"
readonly stop_marker="$run_dir/status/RAY_STOP"
readonly vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
readonly vllm_archive_tag="vllm-0.25.1-nccl-2.28.9"
readonly vllm_archive="$ARCHIVE_ROOT/ray_venvs/$SOURCE_COMMIT/${vllm_actor}.${vllm_archive_tag}.tar"
readonly vllm_archive_marker="${vllm_archive}.success"
readonly gym_archive="$ARCHIVE_ROOT/huggingface/gym_venvs.tar"
readonly gym_archive_marker="${gym_archive}.success"
job_id=${LSB_JOBID:-$$}
ray_launch_pids=()

[[ $TARGET_ENTRYPOINT == "$expected_entrypoint" ]] || { echo "unexpected entrypoint" >&2; exit 2; }
[[ $TARGET_CONFIG == "$expected_config" ]] || { echo "unexpected config: $TARGET_CONFIG" >&2; exit 2; }
[[ $TRAIN_EXPECTED_HOSTS == "$expected_host_count" ]] || { echo "expected 27 hosts" >&2; exit 2; }
[[ $TRAIN_SLOTS_PER_HOST == "$expected_slots_per_host" ]] || { echo "expected 17 slots/host" >&2; exit 2; }

cleanup() {
    local status=$?
    set +e
    touch "$stop_marker"
    local deadline=$((SECONDS + 90))
    while (( SECONDS < deadline )); do
        running=0
        for pid in "${ray_launch_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && running=1 && break
        done
        (( running == 0 )) && break
        sleep 3
    done
    for pid in "${ray_launch_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    deadline=$((SECONDS + 15))
    while (( SECONDS < deadline )); do
        running=0
        for pid in "${ray_launch_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && running=1 && break
        done
        (( running == 0 )) && break
        sleep 1
    done
    for pid in "${ray_launch_pids[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
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
mkdir -p "$run_dir/logs/ray" "$run_dir/status/ray" "$run_dir/wandb" \
    "$run_dir/nemo_rl_logs" "$CHECKPOINT_DIR" "$CACHE_ROOT/megatron_ckpt_cache" \
    "$CACHE_ROOT/hf_config_locks" "$JUDGE_HF_HOME/xet"
rm -f "$run_dir/status/ray"/node-*.ready "$run_dir/status/ray"/node-*.failed

[[ -s "$run_dir/status/PREP_SUCCESS" ]] || { echo "PREP_SUCCESS is missing" >&2; exit 2; }
[[ -s $CONTAINER_IMAGE ]] || { echo "container image is missing" >&2; exit 2; }
[[ -s $vllm_archive && -s $vllm_archive_marker ]] || {
    echo "vLLM worker archive is incomplete: $vllm_archive" >&2
    exit 2
}
[[ -s $gym_archive && -s $gym_archive_marker ]] || { echo "Gym archive is incomplete" >&2; exit 2; }
[[ -s $TRAIN_PATH && -s $VAL_PATH && -d $MODEL_PATH ]] || exit 2
[[ -d $JUDGE_HF_HOME && -w $JUDGE_HF_HOME ]] || { echo "JUDGE_HF_HOME is not writable" >&2; exit 2; }
command -v blaunch >/dev/null || { echo "blaunch is unavailable" >&2; exit 2; }
[[ -n ${LSB_MCPU_HOSTS:-} ]] || { echo "LSB_MCPU_HOSTS is unset" >&2; exit 2; }

read_secret() {
    local name=$1 path=$2 mode value
    [[ -f $path && ! -L $path && -s $path ]] || { echo "$name file is unsafe: $path" >&2; exit 2; }
    mode=$(stat -c %a "$path")
    (( (8#$mode & 077) == 0 )) || { echo "$name file must be mode 600: $path" >&2; exit 2; }
    IFS= read -r value <"$path" || true
    value=${value%$'\r'}
    [[ -n $value ]] || { echo "$name file is empty: $path" >&2; exit 2; }
    printf -v "$name" '%s' "$value"
    export "${name?}"
}
read_secret HF_TOKEN "$HF_TOKEN_FILE"
read_secret WANDB_API_KEY "$WANDB_API_KEY_FILE"
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"

read -r -a host_slot_fields <<<"$LSB_MCPU_HOSTS"
(( ${#host_slot_fields[@]} % 2 == 0 )) || { echo "malformed LSB_MCPU_HOSTS" >&2; exit 2; }
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
    echo "expected ${expected_host_count} hosts, found ${#hosts[@]}" >&2
    exit 2
}

head_host=${hosts[0]}
[[ ${head_host%%.*} == "$(hostname -s)" ]] || {
    echo "LSF driver is not the first allocated host" >&2
    exit 2
}
head_ip=$(ip -4 -o addr show dev "${RAY_INTERFACE:-ibp26s0}" scope global 2>/dev/null \
    | awk '{sub(/\/.*$/, "", $4); print $4; exit}')
head_ip=${head_ip:-$(hostname -I 2>/dev/null | tr ' ' '\n' \
    | awk '/^[0-9]+(\.[0-9]+){3}$/ && $1 !~ /^127\./ {print; exit}')}
[[ -n $head_ip ]] || { echo "unable to resolve Ray head IP" >&2; exit 2; }
head_address="${head_ip}:1200"
printf '%s\n' "${hosts[@]}" >"$run_dir/status/allocated-hosts.txt"
printf 'head_host=%s\nhead_ip=%s\nhead_address=%s\n' "$head_host" "$head_ip" "$head_address" \
    >"$run_dir/status/ray-topology.txt"

printf 'started_at_utc=%s\njob_id=%s\nhost=%s\n' \
    "$(date -u +%FT%TZ)" "$job_id" "$(hostname -f)" >"$run_dir/logs/train-status.log"

launch_ray_node() {
    local host=$1 role=$2 rank=$3
    if [[ ${host%%.*} == "$(hostname -s)" ]]; then
        /bin/bash "$run_dir/control/ray_node.sh" \
            "$run_dir" "$role" "$head_address" "$rank" \
            >"$run_dir/logs/ray/node-${rank}-${host}.log" 2>&1 &
    elif [[ ${NRL_USE_BATTACH_LAUNCHER:-0} == 1 ]]; then
        battach -L "$run_dir/control/attach_ray_node.sh" -m "$host" "$job_id" \
            >"$run_dir/logs/ray/node-${rank}-${host}.log" 2>&1 &
    else
        blaunch -z "$host" /bin/bash "$run_dir/control/ray_node.sh" \
            "$run_dir" "$role" "$head_address" "$rank" \
            >"$run_dir/logs/ray/node-${rank}-${host}.log" 2>&1 &
    fi
    ray_launch_pids+=("$!")
}

wait_for_node_marker() {
    local rank=$1
    local timeout_seconds=$2
    local deadline=$((SECONDS + timeout_seconds))
    while [[ ! -s "$run_dir/status/ray/node-${rank}.ready" ]]; do
        [[ ! -s "$run_dir/status/ray/node-${rank}.failed" ]] || return 1
        (( SECONDS < deadline )) || return 1
        sleep 2
    done
}

launch_ray_node "${hosts[0]}" head 0
wait_for_node_marker 0 1800
for ((rank = 1; rank < expected_host_count; rank++)); do
    launch_ray_node "${hosts[rank]}" worker "$rank"
done
for ((rank = 1; rank < expected_host_count; rank++)); do
    wait_for_node_marker "$rank" 1800 || { echo "Ray node ${rank} failed" >&2; exit 2; }
done
echo "Ray cluster ready: ${expected_host_count} hosts, ${expected_gpu_total} GPUs"

driver_tmp="/tmp/nrl-super-27n-driver-${job_id}"
head_node_tmp="/tmp/nrl-super-27n-ray-${job_id}-0"
mkdir -p "$driver_tmp/ray" "$driver_tmp/torch" "$driver_tmp/triton" \
    "$driver_tmp/hf_modules" "$driver_tmp/uv" "$driver_tmp/xdg" \
    "$driver_tmp/inductor" "$driver_tmp/vllm_compile_cache/deep_gemm" \
    "$driver_tmp/flashinfer_cubins" "$driver_tmp/flashinfer_workspace"
[[ -L $head_node_tmp/vllm_worker/bin/python ]] || { echo "head node-local vLLM environment is missing" >&2; exit 2; }
find "$head_node_tmp/gym_venvs" -mindepth 1 -maxdepth 3 -type d -name .venv -print -quit | grep -q . || {
    echo "head node-local Gym environments are missing" >&2
    exit 2
}
container=(
    "$APPTAINER" exec --fakeroot --nv --contain --writable-tmpfs --no-mount home
    --bind "$SOURCE_DIR:/opt/nemo-rl"
    --bind /proj:/proj
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$head_node_tmp:/job-tmp"
    --bind "$head_node_tmp/gym_venvs:/opt/gym_venvs"
    --bind "$head_node_tmp/vllm_worker:/opt/ray_venvs/$vllm_actor"
    --bind /dev/infiniband:/dev/infiniband
    --bind /etc/resolv.conf:/etc/resolv.conf:ro
    --pwd /opt/nemo-rl
    --env "USER=$(id -un)"
    --env "HF_HOME=$JUDGE_HF_HOME"
    --env "HF_XET_CACHE=$JUDGE_HF_HOME/xet"
    --env HF_MODULES_CACHE=/job-tmp/hf_modules
    --env XDG_CACHE_HOME=/job-tmp/xdg
    --env UV_CACHE_DIR=/job-tmp/uv
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
    --env "TORCH_HOME=/job-tmp/torch"
    --env "TRITON_CACHE_DIR=/job-tmp/triton"
    --env TORCHINDUCTOR_CACHE_DIR=/job-tmp/inductor
    --env VLLM_CACHE_ROOT=/job-tmp/vllm_compile_cache
    --env DG_JIT_CACHE_DIR=/job-tmp/vllm_compile_cache/deep_gemm
    --env FLASHINFER_CUBIN_DIR=/job-tmp/flashinfer_cubins
    --env FLASHINFER_WORKSPACE_BASE=/job-tmp/flashinfer_workspace
    --env "WANDB_DIR=$run_dir/wandb"
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

"${container[@]}" /bin/bash -ce '
cd /opt/nemo-rl
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "$1" \
  "cluster.num_nodes=20" \
  "policy.model_name=$2" \
  "policy.tokenizer.name=$2" \
  "data.train.data_path=$3" \
  "data.validation.data_path=$4" \
  env.nemo_gym.uv_venv_dir=/opt/gym_venvs \
  "env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.hf_home=$8" \
  "env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.hf_home=$8" \
  "env.nemo_gym.genrm_model.responses_api_models.genrm_model.hf_home=$8" \
  "checkpointing.checkpoint_dir=$5" \
  "logger.log_dir=$6/nemo_rl_logs" \
  logger.wandb_enabled=true \
  logger.wandb.project=nemo-rl \
  "logger.wandb.name=$7"
' -- "$TARGET_CONFIG" "$MODEL_PATH" "$TRAIN_PATH" "$VAL_PATH" "$CHECKPOINT_DIR" "$run_dir" "$RUN_ID" "$JUDGE_HF_HOME" \
    2>&1 | tee "$run_dir/logs/train-driver.log"
driver_status=${PIPESTATUS[0]}
printf 'driver_exit_status=%s\n' "$driver_status" >>"$run_dir/logs/train-status.log"
(( driver_status == 0 )) || exit "$driver_status"

wandb_url=$(grep -Eo 'https://wandb\.ai/[^[:space:]]+' "$run_dir/logs/train-driver.log" | tail -n 1 || true)
{
    printf 'completed_at_utc=%s\njob_id=%s\nsource_commit=%s\nsource_dirty=%s\n' \
        "$(date -u +%FT%TZ)" "$job_id" "$SOURCE_COMMIT" "$SOURCE_DIRTY"
    printf 'config=%s\nmodel=%s\ntrain_data=%s\nvalidation_data=%s\n' \
        "$TARGET_CONFIG" "$MODEL_PATH" "$TRAIN_PATH" "$VAL_PATH"
    printf 'ray_hosts=%s\ngpus=%s\nwandb_url=%s\n' "$expected_host_count" "$expected_gpu_total" "${wandb_url:-not-found}"
} >"${success_marker}.tmp"
mv "${success_marker}.tmp" "$success_marker"
echo "Training complete: $success_marker"
