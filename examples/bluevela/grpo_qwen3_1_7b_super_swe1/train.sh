#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

readonly expected_target_test="tests/test_suites/llm/grpo-qwen3-1.7b-1n8g-megatron-super-swe1.sh"
readonly success_marker="$run_dir/status/SUCCESS"
readonly failure_marker="$run_dir/status/TRAIN_FAILED"
readonly vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
readonly vllm_venv_dir="$CACHE_ROOT/ray_venvs/$SOURCE_COMMIT/$vllm_actor"
job_id=${LSB_JOBID:-$$}
node_tmp="/tmp/nrl-super-swe1-train-${job_id}"

[[ $TARGET_TEST == "$expected_target_test" ]] || {
    echo "refusing unexpected target test: ${TARGET_TEST}" >&2
    exit 2
}

case "$node_tmp" in
    /tmp/nrl-super-swe1-train-[0-9]*) ;;
    *)
        echo "refusing unsafe temporary path: ${node_tmp}" >&2
        exit 2
        ;;
esac

cleanup() {
    local status=$?
    printf 'finished_at_utc=%s\nexit_status=%s\n' "$(date -u +%FT%TZ)" "$status" \
        >>"$run_dir/logs/train-status.log"
    if (( status != 0 )); then
        printf 'failed_at_utc=%s\nexit_status=%s\njob_id=%s\n' \
            "$(date -u +%FT%TZ)" "$status" "$job_id" >"${failure_marker}.tmp"
        mv "${failure_marker}.tmp" "$failure_marker"
    fi
    rm -rf -- "$node_tmp"
    exit "$status"
}
trap cleanup EXIT

rm -f "$success_marker" "$failure_marker"
mkdir -p \
    "$node_tmp/ray" "$node_tmp/torch" "$node_tmp/triton" \
    "$run_dir/logs" "$run_dir/status" "$run_dir/home" "$run_dir/wandb" \
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
[[ -s "$CACHE_ROOT/huggingface/superv3_data/swe1/train-split.jsonl" ]]
[[ -s "$CACHE_ROOT/huggingface/superv3_data/swe1/val-split.jsonl" ]]
[[ -s "$CACHE_ROOT/huggingface/gym_venvs/GYM_SUCCESS" ]]
# This absolute Python symlink resolves only after the venv is bind-mounted in
# the container, so the host-side check must test the link itself, not -x.
[[ -s "$vllm_venv_dir/VENV_SUCCESS" && -L "$vllm_venv_dir/bin/python" ]]

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

command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable" >&2
    exit 2
}
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#gpu_names[@]} -eq 8 ]] || {
    echo "expected 8 visible GPUs, found ${#gpu_names[@]}" >&2
    printf '%s\n' "${gpu_names[@]}" >&2
    exit 2
}
for gpu_name in "${gpu_names[@]}"; do
    [[ $gpu_name == *H100* ]] || {
        echo "expected H100, found: ${gpu_name}" >&2
        exit 2
    }
done
nvidia-smi >"$run_dir/logs/nvidia-smi.txt"

source_dir="$run_dir/source"
container=(
    "$APPTAINER" exec
    --fakeroot
    --nv
    --containall
    --writable-tmpfs
    --no-mount home
    --bind "$source_dir:/opt/nemo-rl"
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$node_tmp:/job-tmp"
    --bind "$CACHE_ROOT/huggingface/gym_venvs:/opt/gym_venvs"
    --bind "$vllm_venv_dir:/opt/ray_venvs/$vllm_actor"
    --bind /etc/resolv.conf:/etc/resolv.conf:ro
    --pwd /opt/nemo-rl
    --env USER=yuetai
    --env "HF_HOME=$CACHE_ROOT/huggingface"
    --env "HF_MODULES_CACHE=$CACHE_ROOT/hf_modules"
    --env "XDG_CACHE_HOME=$CACHE_ROOT/xdg"
    --env "UV_CACHE_DIR=$CACHE_ROOT/uv"
    --env UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
    --env VIRTUAL_ENV=/opt/nemo_rl_venv
    --env UV_NO_SYNC=1
    --env UV_PYTHON=/opt/nemo_rl_venv/bin/python
    --env UV_HTTP_TIMEOUT=600
    --env NRL_CONTAINER=1
    --env NRL_IGNORE_VERSION_MISMATCH=1
    --env NRL_WG_USE_RAY_REF=1
    --env NRL_VLLM_USE_V1=1
    --env NEMO_RL_VENV_DIR=/opt/ray_venvs
    --env NEMO_GYM_VENV_DIR=/opt/gym_venvs
    --env "PERSISTENT_CACHE=$CACHE_ROOT"
    --env "RAY_TMPDIR=/job-tmp/ray"
    --env "TMPDIR=/job-tmp"
    --env "TORCH_HOME=/job-tmp/torch"
    --env "TRITON_CACHE_DIR=/job-tmp/triton"
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
    --env PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    "$CONTAINER_IMAGE"
)
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    container=("${container[@]:0:${#container[@]}-1}" --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "$CONTAINER_IMAGE")
fi

driver_log="$run_dir/logs/train-driver.log"
set +e
"${container[@]}" /bin/bash -ce \
    "cd /opt/nemo-rl && bash ${TARGET_TEST}" 2>&1 | tee "$driver_log"
driver_status=${PIPESTATUS[0]}
set -e
printf 'driver_exit_status=%s\n' "$driver_status" >>"$run_dir/logs/train-status.log"
(( driver_status == 0 )) || exit "$driver_status"

experiment_name=$(basename "$TARGET_TEST" .sh)
metrics_file="$source_dir/tests/test_suites/llm/$experiment_name/metrics.json"
run_log="$source_dir/tests/test_suites/llm/$experiment_name/run.log"
[[ -s $metrics_file ]] || {
    echo "metrics file is missing: ${metrics_file}" >&2
    exit 2
}
jq -e '."train/loss" | has("5")' "$metrics_file" >/dev/null || {
    echo "training driver exited zero without reaching step 5" >&2
    exit 2
}
[[ -s $run_log ]] || {
    echo "nightly run log is missing: ${run_log}" >&2
    exit 2
}

metrics_summary=$(jq -c '{
  step5_loss: ."train/loss"["5"],
  step5_token_mult_prob_error: ."train/token_mult_prob_error"["5"],
  token_mult_prob_error: ."train/token_mult_prob_error",
  gen_kl_error: ."train/gen_kl_error"
}' "$metrics_file")
wandb_url=$(grep -Eo 'https://wandb\.ai/[^[:space:]]+' "$driver_log" | tail -n 1 || true)

{
    printf 'completed_at_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'job_id=%s\n' "$job_id"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'target_test=%s\n' "$TARGET_TEST"
    printf 'metrics_file=%s\n' "$metrics_file"
    printf 'metrics_summary=%s\n' "$metrics_summary"
    printf 'wandb_url=%s\n' "${wandb_url:-not-found-in-console-log}"
} >"${success_marker}.tmp"
mv "${success_marker}.tmp" "$success_marker"

echo "Training complete: ${success_marker}"
