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

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd)"

readonly target_entrypoint="examples/nemo_gym/run_grpo_nemo_gym.py"
readonly target_config="examples/nemo_gym/grpo_nanov3.yaml"
readonly model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
readonly tokenizer_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
readonly default_storage_root="/proj/datasets/interns/yuetai/agent_envs/more_task"

usage() {
    cat <<'EOF'
Usage: submit.sh [--dry-run|--submit]

  --dry-run  Print the resolved preparation and training submissions.
             This is the default and does not create the run directory.
  --submit   Create a unique run directory, submit preparation, then submit a
             dependent Nano v3 GRPO training job.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

validate_identifier() {
    local name=$1
    local value=$2
    [[ $value =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || die "${name} contains unsupported characters: ${value}"
}

require_proj_path() {
    local name=$1
    local value=$2
    [[ $value == /proj/* ]] \
        || die "${name} must be an absolute /proj path: ${value}"
}

validate_secret_file() {
    local name=$1
    local path=$2
    local mode owner
    [[ -f $path && ! -L $path && -s $path ]] \
        || die "${name} must be a non-empty regular file, not a symlink: ${path}"
    owner=$(stat -c %u "$path")
    [[ $owner == "$(id -u)" ]] \
        || die "${name} must be owned by $(id -un): ${path}"
    mode=$(stat -c %a "$path")
    (( (8#$mode & 077) == 0 )) \
        || die "${name} must not be accessible by group/other (use chmod 600): ${path}"
}

write_export() {
    local name=$1
    local value=$2
    printf 'export %s=%q\n' "$name" "$value"
}

parse_job_id() {
    sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' <<<"$1" | head -n 1
}

mode=${1:---dry-run}
[[ $# -le 1 ]] || {
    usage >&2
    exit 2
}
case "$mode" in
    --dry-run | --submit) ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        die "unsupported mode: ${mode}"
        ;;
esac

run_root=${RUN_ROOT:-${default_storage_root}/runs}
cache_root=${CACHE_ROOT:-${default_storage_root}/cache/nemo-rl-nanov3-32n}
container_image=${CONTAINER_IMAGE:-${default_storage_root}/images/nemo-rl-v0.7.0.sif}
apptainer=${APPTAINER:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
source_repo=${SOURCE_REPO:-https://github.com/NVIDIA-NeMo/RL.git}
source_commit=${SOURCE_COMMIT:-$(git -C "$repo_dir" rev-parse HEAD)}
run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-grpo-nanov3-32n8g-yuetai}
run_dir=${RUN_DIR:-${run_root}/nemo-rl-nanov3-32n/${run_id}}
checkpoint_dir=${CHECKPOINT_DIR:-${run_dir}/checkpoints}
wandb_key_file=${WANDB_API_KEY_FILE:-${HOME:?}/.config/megatron-bridge/wandb_api_key}
hf_token_file=${HF_TOKEN_FILE:-${HOME:?}/.cache/huggingface/token}

lsf_queue=${LSF_QUEUE:-priority}
lsf_group=${LSF_GROUP:-grp_models}
prep_slots=${PREP_SLOTS:-64}
prep_memory_mb=${PREP_MEMORY_MB:-524288}
prep_walltime=${PREP_WALLTIME:-06:00}
prep_tmp_mb=${PREP_TMP_MB:-153600}
train_hosts=${TRAIN_HOSTS:-32}
train_slots_per_host=17
train_slots=${TRAIN_SLOTS:-$((train_hosts * train_slots_per_host))}
train_memory_mb=${TRAIN_MEMORY_MB:-524288}
train_walltime=${TRAIN_WALLTIME:-04:00}
gpu_requirement=${GPU_REQUIREMENT:-num=8:mode=shared:j_exclusive=yes}
prep_job_name=${PREP_JOB_NAME:-nrl-nanov3-32n-prep-yuetai}
train_job_name=${TRAIN_JOB_NAME:-nrl-nanov3-32n-train-yuetai}

validate_identifier RUN_ID "$run_id"
validate_identifier PREP_JOB_NAME "$prep_job_name"
validate_identifier TRAIN_JOB_NAME "$train_job_name"
[[ $source_commit =~ ^[0-9a-fA-F]{40}$ ]] \
    || die "SOURCE_COMMIT must be a full 40-character Git commit: ${source_commit}"
[[ $lsf_queue == priority ]] || die "LSF_QUEUE must be priority for this example"
[[ $lsf_group == grp_models ]] || die "LSF_GROUP must be grp_models for this example"
[[ $prep_slots == 64 ]] || die "PREP_SLOTS must remain 64"
[[ $train_hosts =~ ^[0-9]+$ && $train_hosts -ge 1 ]] || die "TRAIN_HOSTS must be positive"
[[ $train_slots == $((train_hosts * train_slots_per_host)) ]] \
    || die "TRAIN_SLOTS must equal TRAIN_HOSTS * ${train_slots_per_host}"
[[ $gpu_requirement == num=8:mode=shared:j_exclusive=yes ]] \
    || die "GPU_REQUIREMENT must reserve all eight GPUs per host"

prep_resource="span[hosts=1] select[tmp>${prep_tmp_mb}] rusage[mem=${prep_memory_mb}]"
train_resource="span[ptile=${train_slots_per_host}] rusage[mem=${train_memory_mb}]"
prep_payload="${run_dir}/control/prepare.sh"
train_payload="${run_dir}/control/train.sh"

render() {
    local dependency=$1
    echo "Run ID: ${run_id}"
    echo "Run directory: ${run_dir}"
    echo "Checkpoint directory: ${checkpoint_dir}"
    echo "Cache root: ${cache_root}"
    echo "Container image: ${container_image}"
    echo "Source: ${source_repo} @ ${source_commit}"
    echo "Entrypoint: ${target_entrypoint}"
    echo "Config: ${target_config}"
    echo "Model: ${model_id}"
    echo "Tokenizer: ${tokenizer_id}"
    echo "Training shape: ${train_hosts} hosts x 8 GPUs = $((train_hosts * 8)) GPUs"
    echo "Preparation resources: -q ${lsf_queue} -G ${lsf_group} -n ${prep_slots} -M ${prep_memory_mb} -W ${prep_walltime} -R ${prep_resource}"
    echo "Training resources: -q ${lsf_queue} -G ${lsf_group} -n ${train_slots} -M ${train_memory_mb} -W ${train_walltime} -R ${train_resource} -gpu ${gpu_requirement}"
    echo "Training dependency: ${dependency}"
    echo "Preparation submission:"
    print_command bsub \
        -q "$lsf_queue" -G "$lsf_group" -n "$prep_slots" \
        -M "$prep_memory_mb" -W "$prep_walltime" -R "$prep_resource" \
        -J "$prep_job_name" \
        -o "${run_dir}/logs/prep.%J.out" -e "${run_dir}/logs/prep.%J.err" \
        /bin/bash "$prep_payload" "$run_dir"
    echo "Training submission:"
    print_command bsub \
        -q "$lsf_queue" -G "$lsf_group" -n "$train_slots" \
        -M "$train_memory_mb" -W "$train_walltime" -R "$train_resource" \
        -gpu "$gpu_requirement" -w "$dependency" -J "$train_job_name" \
        -o "${run_dir}/logs/train.%J.out" -e "${run_dir}/logs/train.%J.err" \
        /bin/bash "$train_payload" "$run_dir"
}

if [[ $mode == --dry-run ]]; then
    render 'done(PREP_JOB_ID)'
    [[ ! -e $run_dir ]] \
        || echo "WARNING: dry-run did not modify state, but RUN_DIR already exists: ${run_dir}" >&2
    exit 0
fi

require_proj_path RUN_ROOT "$run_root"
require_proj_path RUN_DIR "$run_dir"
require_proj_path CHECKPOINT_DIR "$checkpoint_dir"
require_proj_path CACHE_ROOT "$cache_root"
require_proj_path CONTAINER_IMAGE "$container_image"
[[ ! -e $run_dir ]] || die "refusing to reuse an existing RUN_DIR: ${run_dir}"
[[ -x $apptainer ]] || die "APPTAINER is not executable: ${apptainer}"
command -v bsub >/dev/null || die "bsub is unavailable"
command -v bjobs >/dev/null || die "bjobs is unavailable"
command -v blaunch >/dev/null || die "blaunch is unavailable"
command -v git >/dev/null || die "git is unavailable"
git -C "$repo_dir" cat-file -e "${source_commit}:${target_entrypoint}" \
    || die "entrypoint does not exist in SOURCE_COMMIT ${source_commit}"
git -C "$repo_dir" cat-file -e "${source_commit}:${target_config}" \
    || die "config does not exist in SOURCE_COMMIT ${source_commit}"
validate_secret_file WANDB_API_KEY_FILE "$wandb_key_file"
validate_secret_file HF_TOKEN_FILE "$hf_token_file"

mkdir -p \
    "$run_dir/control" "$run_dir/logs/ray" "$run_dir/status/ray" \
    "$run_dir/tmp" "$run_dir/home" "$run_dir/wandb" \
    "$checkpoint_dir" "$run_dir/nemo_rl_logs" \
    "$cache_root" "$(dirname -- "$container_image")"
cp "$script_dir/prepare.sh" "$script_dir/train.sh" "$script_dir/ray_node.sh" \
    "$run_dir/control/"
chmod 700 \
    "$run_dir/control/prepare.sh" "$run_dir/control/train.sh" \
    "$run_dir/control/ray_node.sh"

{
    write_export RUN_ID "$run_id"
    write_export RUN_DIR "$run_dir"
    write_export CHECKPOINT_DIR "$checkpoint_dir"
    write_export CACHE_ROOT "$cache_root"
    write_export CONTAINER_IMAGE "$container_image"
    write_export APPTAINER "$apptainer"
    write_export SOURCE_REPO "$source_repo"
    write_export SOURCE_COMMIT "$source_commit"
    write_export WANDB_API_KEY_FILE "$wandb_key_file"
    write_export HF_TOKEN_FILE "$hf_token_file"
    write_export TARGET_ENTRYPOINT "$target_entrypoint"
    write_export TARGET_CONFIG "$target_config"
    write_export MODEL_ID "$model_id"
    write_export TOKENIZER_ID "$tokenizer_id"
    write_export TRAIN_EXPECTED_HOSTS "$train_hosts"
    write_export TRAIN_SLOTS_PER_HOST "$train_slots_per_host"
    write_export RAY_EXPECTED_HOSTS "$train_hosts"
} >"$run_dir/control/run.env"
chmod 600 "$run_dir/control/run.env"
sha256sum \
    "$run_dir/control/prepare.sh" "$run_dir/control/train.sh" \
    "$run_dir/control/ray_node.sh" >"$run_dir/control/payloads.sha256"

{
    printf 'run_id=%s\n' "$run_id"
    printf 'run_dir=%s\n' "$run_dir"
    printf 'checkpoint_dir=%s\n' "$checkpoint_dir"
    printf 'submitted_at_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'source_repo=%s\n' "$source_repo"
    printf 'source_commit=%s\n' "$source_commit"
    printf 'container_image=%s\n' "$container_image"
    printf 'target_entrypoint=%s\n' "$target_entrypoint"
    printf 'target_config=%s\n' "$target_config"
    printf 'model_id=%s\n' "$model_id"
    printf 'tokenizer_id=%s\n' "$tokenizer_id"
    printf 'training_shape=%s hosts x 8 GPUs\n' "$train_hosts"
    printf 'queue=%s\n' "$lsf_queue"
    printf 'group=%s\n' "$lsf_group"
    printf 'expected_prep_marker=%s/status/PREP_SUCCESS\n' "$run_dir"
    printf 'expected_success_marker=%s/status/SUCCESS\n' "$run_dir"
} >"$run_dir/RUN_INFO.txt"

prep_output=$(bsub \
    -q "$lsf_queue" -G "$lsf_group" -n "$prep_slots" \
    -M "$prep_memory_mb" -W "$prep_walltime" -R "$prep_resource" \
    -J "$prep_job_name" \
    -o "${run_dir}/logs/prep.%J.out" -e "${run_dir}/logs/prep.%J.err" \
    /bin/bash "$run_dir/control/prepare.sh" "$run_dir")
prep_job_id=$(parse_job_id "$prep_output")
[[ -n $prep_job_id ]] \
    || die "unable to parse preparation job ID from: ${prep_output}"

dependency="done(${prep_job_id})"
train_output=$(bsub \
    -q "$lsf_queue" -G "$lsf_group" -n "$train_slots" \
    -M "$train_memory_mb" -W "$train_walltime" -R "$train_resource" \
    -gpu "$gpu_requirement" -w "$dependency" -J "$train_job_name" \
    -o "${run_dir}/logs/train.%J.out" -e "${run_dir}/logs/train.%J.err" \
    /bin/bash "$run_dir/control/train.sh" "$run_dir")
train_job_id=$(parse_job_id "$train_output")
[[ -n $train_job_id ]] \
    || die "unable to parse training job ID from: ${train_output}"

{
    printf 'prep_job_id=%s\n' "$prep_job_id"
    printf 'train_job_id=%s\n' "$train_job_id"
    printf 'train_dependency=%s\n' "$dependency"
} >>"$run_dir/RUN_INFO.txt"

echo "$prep_output"
echo "$train_output"
echo "Run directory: ${run_dir}"
echo "Preparation job: ${prep_job_id}"
echo "Training job: ${train_job_id} (dependency: ${dependency})"
echo "Monitor: bjobs -l ${prep_job_id} ${train_job_id}"
