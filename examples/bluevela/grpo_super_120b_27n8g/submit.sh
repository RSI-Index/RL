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
readonly default_target_config="examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml"
readonly default_storage_root="/proj/datasets/interns/yuetai/agent_envs/more_task"
readonly default_cache_root="/u/yuetai/more_task/cache/nemo-rl-super-27n"
readonly default_archive_root="${default_storage_root}/cache/nemo-rl-super-27n-archives"

usage() {
    cat <<'EOF'
Usage: submit.sh [--dry-run|--setup-only|--submit]

Required environment variables:
  MODEL_PATH, TRAIN_PATH, VAL_PATH

Optional environment variables:
  SOURCE_DIR, TARGET_CONFIG, CONTAINER_IMAGE, RUN_ROOT, CACHE_ROOT, ARCHIVE_ROOT,
  RUN_DIR, CHECKPOINT_DIR, RUN_ID, LSF_QUEUE, TRAIN_EXCLUDE_HOSTS
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
    local name=$1 value=$2
    [[ $value =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || die "${name} contains unsupported characters: ${value}"
}

require_proj_path() {
    local name=$1 value=$2
    [[ $value == /proj/* ]] || die "${name} must be an absolute /proj path: ${value}"
}

require_cache_path() {
    local name=$1 value=$2
    [[ $value == /proj/datasets/interns/yuetai/* || $value == /u/yuetai/* ]] \
        || die "${name} must be under /proj/datasets/interns/yuetai or /u/yuetai: ${value}"
}

validate_secret_file() {
    local name=$1 path=$2 mode owner
    [[ -f $path && ! -L $path && -s $path ]] \
        || die "${name} must be a non-empty regular file: ${path}"
    owner=$(stat -c %u "$path")
    [[ $owner == "$(id -u)" ]] || die "${name} must be owned by $(id -un): ${path}"
    mode=$(stat -c %a "$path")
    (( (8#$mode & 077) == 0 )) || die "${name} must be mode 600: ${path}"
}

write_export() {
    printf 'export %s=%q\n' "$1" "$2"
}

parse_job_id() {
    sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' <<<"$1" | head -n 1
}

mode=${1:---dry-run}
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "$mode" in
    --dry-run|--setup-only|--submit) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unsupported mode: ${mode}" ;;
esac

model_path=${MODEL_PATH:?MODEL_PATH is required}
train_path=${TRAIN_PATH:?TRAIN_PATH is required}
val_path=${VAL_PATH:?VAL_PATH is required}
source_dir=${SOURCE_DIR:-$repo_dir}
target_config=${TARGET_CONFIG:-$default_target_config}
run_root=${RUN_ROOT:-${default_storage_root}/runs}
cache_root=${CACHE_ROOT:-$default_cache_root}
archive_root=${ARCHIVE_ROOT:-$default_archive_root}
container_image=${CONTAINER_IMAGE:-${default_storage_root}/images/nemo-rl-v0.7.0.sif}
apptainer=${APPTAINER:-/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer}
run_id=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-grpo-super-27n8g-yuetai}
run_dir=${RUN_DIR:-${run_root}/nemo-rl-super-27n/${run_id}}
checkpoint_dir=${CHECKPOINT_DIR:-${run_dir}/checkpoints}
wandb_key_file=${WANDB_API_KEY_FILE:-${HOME:?}/.config/megatron-bridge/wandb_api_key}
hf_token_file=${HF_TOKEN_FILE:-${HOME:?}/.cache/huggingface/token}

lsf_queue=${LSF_QUEUE:-priority}
lsf_group=${LSF_GROUP:-grp_models}
prep_slots=64
prep_memory_mb=524288
prep_walltime=${PREP_WALLTIME:-06:00}
prep_tmp_mb=153600
train_hosts=27
train_slots_per_host=17
train_slots=$((train_hosts * train_slots_per_host))
train_memory_mb=524288
train_walltime=${TRAIN_WALLTIME:-04:00}
gpu_requirement=${GPU_REQUIREMENT:-num=8:mode=shared:j_exclusive=yes}
train_exclude_hosts=${TRAIN_EXCLUDE_HOSTS:-}

validate_identifier RUN_ID "$run_id"
[[ $lsf_queue == priority ]] || die "LSF_QUEUE must remain priority for this launcher"
[[ $lsf_group == grp_models ]] || die "LSF_GROUP must remain grp_models"
[[ $gpu_requirement == num=8:mode=shared:j_exclusive=yes ]] \
    || die "GPU_REQUIREMENT must reserve eight job-exclusive H100 GPUs"
[[ -d $source_dir/.git ]] || die "SOURCE_DIR is not a Git checkout: ${source_dir}"
[[ -f $source_dir/$target_entrypoint ]] || die "entrypoint is missing: ${source_dir}/${target_entrypoint}"
[[ -f $source_dir/$target_config ]] || die "config is missing: ${source_dir}/${target_config}"

prep_resource="span[hosts=1] select[tmp>${prep_tmp_mb}] rusage[mem=${prep_memory_mb}]"
train_resource="span[ptile=${train_slots_per_host}] rusage[mem=${train_memory_mb}]"
if [[ -n $train_exclude_hosts ]]; then
    read -r -a excluded_hosts <<<"$train_exclude_hosts"
    (( ${#excluded_hosts[@]} <= 16 )) || die "TRAIN_EXCLUDE_HOSTS supports at most 16 hosts"
    terms=()
    for host in "${excluded_hosts[@]}"; do
        [[ $host =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || die "invalid excluded host: ${host}"
        terms+=("hname!='${host}'")
    done
    train_select=$(IFS=' && '; echo "${terms[*]}")
    train_resource="select[${train_select}] ${train_resource}"
fi

prep_payload="${run_dir}/control/prepare.sh"
train_payload="${run_dir}/control/train.sh"

render() {
    local dependency=$1
    echo "Run ID: ${run_id}"
    echo "Run directory: ${run_dir}"
    echo "Source: ${source_dir}"
    echo "Config: ${target_config}"
    echo "Model: ${model_path}"
    echo "Train data: ${train_path}"
    echo "Val data: ${val_path}"
    echo "Ray hosts: ${train_hosts} x 8 GPUs; NeMo-RL cluster.num_nodes=20"
    echo "CPU/memory: ${train_slots_per_host} slots, $((train_memory_mb / 1024)) GiB per host"
    echo "Preparation: -q ${lsf_queue} -G ${lsf_group} -n ${prep_slots} -M ${prep_memory_mb} -W ${prep_walltime} -R ${prep_resource}"
    echo "Training: -q ${lsf_queue} -G ${lsf_group} -n ${train_slots} -M ${train_memory_mb} -W ${train_walltime} -R ${train_resource} -gpu ${gpu_requirement}"
    echo "Dependency: ${dependency}"
    print_command bsub -q "$lsf_queue" -G "$lsf_group" -n "$prep_slots" \
        -M "$prep_memory_mb" -W "$prep_walltime" -R "$prep_resource" \
        -J "nrl-super-27n-prep-yuetai" \
        -o "${run_dir}/logs/prep.%J.out" -e "${run_dir}/logs/prep.%J.err" \
        /bin/bash "$prep_payload" "$run_dir"
    print_command bsub -q "$lsf_queue" -G "$lsf_group" -n "$train_slots" \
        -M "$train_memory_mb" -W "$train_walltime" -R "$train_resource" \
        -gpu "$gpu_requirement" -w "$dependency" \
        -J "nrl-super-27n-train-yuetai" \
        -o "${run_dir}/logs/train.%J.out" -e "${run_dir}/logs/train.%J.err" \
        /bin/bash "$train_payload" "$run_dir"
}

if [[ $mode == --dry-run ]]; then
    render 'done(PREP_JOB_ID)'
    exit 0
fi

require_proj_path RUN_ROOT "$run_root"
require_proj_path RUN_DIR "$run_dir"
require_cache_path CACHE_ROOT "$cache_root"
require_proj_path ARCHIVE_ROOT "$archive_root"
require_proj_path CHECKPOINT_DIR "$checkpoint_dir"
require_proj_path MODEL_PATH "$model_path"
require_proj_path TRAIN_PATH "$train_path"
require_proj_path VAL_PATH "$val_path"
[[ -d $model_path ]] || die "MODEL_PATH is not a directory: ${model_path}"
[[ -s $train_path ]] || die "TRAIN_PATH is missing or empty: ${train_path}"
[[ -s $val_path ]] || die "VAL_PATH is missing or empty: ${val_path}"
[[ -x $apptainer ]] || die "APPTAINER is not executable: ${apptainer}"
[[ -s $container_image ]] || die "CONTAINER_IMAGE is missing or empty: ${container_image}"
command -v bsub >/dev/null || die "bsub is unavailable"
command -v bjobs >/dev/null || die "bjobs is unavailable"
command -v blaunch >/dev/null || die "blaunch is unavailable"
validate_secret_file WANDB_API_KEY_FILE "$wandb_key_file"
validate_secret_file HF_TOKEN_FILE "$hf_token_file"
[[ ! -e $run_dir ]] || die "refusing to reuse RUN_DIR: ${run_dir}"

mkdir -p "$run_dir/control" "$run_dir/logs/ray" "$run_dir/status/ray" \
    "$run_dir/tmp" "$run_dir/home" "$run_dir/wandb" "$run_dir/nemo_rl_logs" \
    "$run_dir/status" "$checkpoint_dir" "$cache_root" "$archive_root"
cp "$script_dir/prepare.sh" "$script_dir/train.sh" "$script_dir/ray_node.sh" \
    "$run_dir/control/"
chmod 700 "$run_dir/control/prepare.sh" "$run_dir/control/train.sh" "$run_dir/control/ray_node.sh"

source_commit=$(git -C "$source_dir" rev-parse HEAD)
source_dirty=false
git -C "$source_dir" diff --quiet && git -C "$source_dir" diff --cached --quiet || source_dirty=true
{
    write_export RUN_ID "$run_id"
    write_export RUN_DIR "$run_dir"
    write_export SOURCE_DIR "$source_dir"
    write_export SOURCE_COMMIT "$source_commit"
    write_export SOURCE_DIRTY "$source_dirty"
    write_export TARGET_ENTRYPOINT "$target_entrypoint"
    write_export TARGET_CONFIG "$target_config"
    write_export MODEL_PATH "$model_path"
    write_export TRAIN_PATH "$train_path"
    write_export VAL_PATH "$val_path"
    write_export CHECKPOINT_DIR "$checkpoint_dir"
    write_export CACHE_ROOT "$cache_root"
    write_export ARCHIVE_ROOT "$archive_root"
    write_export CONTAINER_IMAGE "$container_image"
    write_export APPTAINER "$apptainer"
    write_export WANDB_API_KEY_FILE "$wandb_key_file"
    write_export HF_TOKEN_FILE "$hf_token_file"
    write_export TRAIN_EXPECTED_HOSTS "$train_hosts"
    write_export TRAIN_SLOTS_PER_HOST "$train_slots_per_host"
    write_export RAY_EXPECTED_HOSTS "$train_hosts"
} >"$run_dir/control/run.env"
chmod 600 "$run_dir/control/run.env"
sha256sum "$run_dir/control/prepare.sh" "$run_dir/control/train.sh" "$run_dir/control/ray_node.sh" \
    >"$run_dir/control/payloads.sha256"
{
    printf 'run_id=%s\nrun_dir=%s\nsubmitted_at_utc=%s\n' "$run_id" "$run_dir" "$(date -u +%FT%TZ)"
    printf 'source_dir=%s\nsource_commit=%s\nsource_dirty=%s\n' "$source_dir" "$source_commit" "$source_dirty"
    printf 'config=%s\nmodel=%s\ntrain_data=%s\nvalidation_data=%s\n' "$target_config" "$model_path" "$train_path" "$val_path"
    printf 'container_image=%s\nqueue=%s\ngroup=%s\nray_hosts=%s\n' "$container_image" "$lsf_queue" "$lsf_group" "$train_hosts"
} >"$run_dir/RUN_INFO.txt"

if [[ $mode == --setup-only ]]; then
    echo "Setup complete; no jobs were submitted."
    echo "Run directory: ${run_dir}"
    echo "Prepare in the current allocation: /bin/bash ${prep_payload} ${run_dir}"
    echo "Train in the current allocation: /bin/bash ${train_payload} ${run_dir}"
    exit 0
fi

prep_output=$(bsub -q "$lsf_queue" -G "$lsf_group" -n "$prep_slots" \
    -M "$prep_memory_mb" -W "$prep_walltime" -R "$prep_resource" \
    -J "nrl-super-27n-prep-yuetai" \
    -o "${run_dir}/logs/prep.%J.out" -e "${run_dir}/logs/prep.%J.err" \
    /bin/bash "$run_dir/control/prepare.sh" "$run_dir")
prep_job_id=$(parse_job_id "$prep_output")
[[ -n $prep_job_id ]] || die "unable to parse preparation job ID: ${prep_output}"

train_output=$(bsub -q "$lsf_queue" -G "$lsf_group" -n "$train_slots" \
    -M "$train_memory_mb" -W "$train_walltime" -R "$train_resource" \
    -gpu "$gpu_requirement" -w "done(${prep_job_id})" \
    -J "nrl-super-27n-train-yuetai" \
    -o "${run_dir}/logs/train.%J.out" -e "${run_dir}/logs/train.%J.err" \
    /bin/bash "$run_dir/control/train.sh" "$run_dir")
train_job_id=$(parse_job_id "$train_output")
[[ -n $train_job_id ]] || die "unable to parse training job ID: ${train_output}"

printf 'prep_job_id=%s\ntrain_job_id=%s\n' "$prep_job_id" "$train_job_id" >>"$run_dir/RUN_INFO.txt"
echo "$prep_output"
echo "$train_output"
echo "Run directory: ${run_dir}"
