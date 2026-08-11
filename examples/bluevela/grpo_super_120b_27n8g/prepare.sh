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

run_dir=${1:?usage: prepare.sh RUN_DIR}
source "$run_dir/control/run.env"

readonly prep_marker="$run_dir/status/PREP_SUCCESS"
readonly failure_marker="$run_dir/status/PREP_FAILED"
readonly vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
readonly vllm_venv_dir="$CACHE_ROOT/ray_venvs/$SOURCE_COMMIT/$vllm_actor"
readonly vllm_venv_marker="$vllm_venv_dir/VENV_SUCCESS"
readonly vllm_archive_tag="vllm-0.25.1-nccl-2.28.9"
readonly vllm_archive="$ARCHIVE_ROOT/ray_venvs/$SOURCE_COMMIT/${vllm_actor}.${vllm_archive_tag}.tar"
readonly vllm_archive_marker="${vllm_archive}.success"
readonly vllm_legacy_archive="$ARCHIVE_ROOT/ray_venvs/$SOURCE_COMMIT/${vllm_actor}.tar"
readonly vllm_legacy_archive_marker="${vllm_legacy_archive}.success"
readonly gym_archive="$ARCHIVE_ROOT/huggingface/gym_venvs.tar"
readonly gym_marker="${gym_archive}.success"
job_id=${LSB_JOBID:-$$}
node_tmp="/tmp/nrl-super-27n-prep-${job_id}"
readonly vllm_local_build="$node_tmp/vllm-worker"
readonly gym_local_build="$node_tmp/gym-venvs"

cleanup() {
    local status=$?
    printf 'finished_at_utc=%s\nexit_status=%s\n' "$(date -u +%FT%TZ)" "$status" \
        >>"$run_dir/logs/prep-status.log"
    if (( status != 0 )); then
        printf 'failed_at_utc=%s\nexit_status=%s\njob_id=%s\n' \
            "$(date -u +%FT%TZ)" "$status" "$job_id" >"${failure_marker}.tmp"
        mv "${failure_marker}.tmp" "$failure_marker"
    fi
    rm -rf -- "$node_tmp"
    exit "$status"
}
trap cleanup EXIT

rm -f "$prep_marker" "$failure_marker"
mkdir -p "$node_tmp/apptainer-tmp" "$node_tmp/ray" "$gym_local_build" \
    "$run_dir/logs" "$run_dir/status" \
    "$CACHE_ROOT/apptainer-cache" "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_modules" \
    "$CACHE_ROOT/uv" "$CACHE_ROOT/xdg" "$CACHE_ROOT/torch" "$CACHE_ROOT/triton" \
    "$CACHE_ROOT/inductor" "$CACHE_ROOT/vllm_compile_cache" "$CACHE_ROOT/flashinfer_cubins" \
    "$CACHE_ROOT/flashinfer_workspace" "$(dirname -- "$CONTAINER_IMAGE")" \
    "$(dirname -- "$vllm_venv_dir")" "$(dirname -- "$vllm_archive")" \
    "$(dirname -- "$gym_archive")"

[[ -x $APPTAINER ]] || { echo "Apptainer is not executable: $APPTAINER" >&2; exit 2; }
[[ -s $CONTAINER_IMAGE ]] || { echo "container image is missing: $CONTAINER_IMAGE" >&2; exit 2; }
[[ -d $SOURCE_DIR/.git ]] || { echo "SOURCE_DIR is not a Git checkout: $SOURCE_DIR" >&2; exit 2; }
[[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$SOURCE_COMMIT" ]] || {
    echo "SOURCE_DIR moved away from recorded commit $SOURCE_COMMIT" >&2
    exit 2
}
[[ -f "$SOURCE_DIR/$TARGET_ENTRYPOINT" ]] || { echo "entrypoint is missing" >&2; exit 2; }
[[ -f "$SOURCE_DIR/$TARGET_CONFIG" ]] || { echo "config is missing" >&2; exit 2; }
[[ -d $MODEL_PATH && -s $MODEL_PATH/config.json ]] || {
    echo "MODEL_PATH must contain config.json: $MODEL_PATH" >&2
    exit 2
}
[[ -s $TRAIN_PATH ]] || { echo "TRAIN_PATH is missing or empty: $TRAIN_PATH" >&2; exit 2; }
[[ -s $VAL_PATH ]] || { echo "VAL_PATH is missing or empty: $VAL_PATH" >&2; exit 2; }

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
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

container=(
    "$APPTAINER" exec --fakeroot --containall --writable-tmpfs --no-mount home
    --bind "$SOURCE_DIR:/opt/nemo-rl"
    --bind /proj:/proj
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$node_tmp:/job-tmp"
    --bind "$gym_local_build:/opt/gym_venvs"
    --bind /etc/resolv.conf:/etc/resolv.conf:ro
    --pwd /opt/nemo-rl
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
    --env NEMO_GYM_VENV_DIR=/opt/gym_venvs
    --env "PERSISTENT_CACHE=$CACHE_ROOT"
    --env RAY_TMPDIR=/job-tmp/ray
    --env TMPDIR=/job-tmp
    --env "TORCH_HOME=$CACHE_ROOT/torch"
    --env "TRITON_CACHE_DIR=$CACHE_ROOT/triton"
    --env "TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor"
    --env "VLLM_CACHE_ROOT=$CACHE_ROOT/vllm_compile_cache"
    --env "FLASHINFER_CUBIN_DIR=$CACHE_ROOT/flashinfer_cubins"
    --env "FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer_workspace"
    --env PATH=/opt/nemo_rl_venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    "$CONTAINER_IMAGE"
)

"${container[@]}" /bin/bash -ce '
test -x /opt/nemo_rl_venv/bin/python
command -v uv >/dev/null
python -c "import nemo_rl, ray, torch; print(nemo_rl.__version__, ray.__version__, torch.__version__)"
python - "$1" <<"PY"
from transformers import AutoConfig
import sys
AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print("model config validated")
PY
' -- "$MODEL_PATH"

if [[ ! -s $vllm_archive_marker ]]; then
    vllm_archive_build="${vllm_archive}.build.${job_id}"
    [[ ! -e $vllm_archive ]] || { echo "vLLM archive exists without marker: $vllm_archive" >&2; exit 2; }
    [[ ! -e $vllm_archive_build ]] || { echo "stale vLLM archive build exists: $vllm_archive_build" >&2; exit 2; }
    if [[ -s $vllm_legacy_archive && -s $vllm_legacy_archive_marker ]]; then
        mkdir -p "$vllm_local_build"
        tar -C "$vllm_local_build" -xf "$vllm_legacy_archive"
    elif [[ -s $vllm_venv_marker && -L $vllm_venv_dir/bin/python ]]; then
        mkdir -p "$vllm_local_build"
        cp -a --reflink=never "$vllm_venv_dir/." "$vllm_local_build/"
    else
        "${container[@]}" /bin/bash -ce '
unset UV_NO_SYNC UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
src=/opt/ray_venvs/$1
dst=/job-tmp/vllm-worker
[[ -d $src ]] || { echo "base Ray vLLM environment is missing: $src" >&2; exit 2; }
mkdir -p "$dst" /job-tmp/uv
cp -a --reflink=never "$src/." "$dst/"
ray_version=$("$dst/bin/python" -c "import importlib.metadata; print(importlib.metadata.version(\"ray\"))")
torch_version=$("$dst/bin/python" -c "import importlib.metadata; print(importlib.metadata.version(\"torch\"))")
UV_CACHE_DIR=/job-tmp/uv uv pip install --python "$dst/bin/python" \
  --index https://flashinfer.ai/whl/cu130 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "flashinfer-jit-cache==0.6.13+cu130"
PYTHONPATH=/opt/nemo-rl "$dst/bin/python" - "$ray_version" "$torch_version" <<"PY"
import importlib.metadata, sys
assert importlib.metadata.version("vllm") == "0.25.1"
assert importlib.metadata.version("flashinfer-jit-cache") == "0.6.13+cu130"
assert importlib.metadata.version("ray") == sys.argv[1]
assert importlib.metadata.version("torch") == sys.argv[2]
PY
' -- "$vllm_actor"
    fi
    [[ -L $vllm_local_build/bin/python ]] || { echo "local vLLM build is incomplete" >&2; exit 2; }
    "${container[@]}" /bin/bash -ce '
unset UV_NO_SYNC UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
dst=/job-tmp/vllm-worker
uv pip install --no-config --python "$dst/bin/python" --reinstall "nvidia-nccl-cu13==2.28.9"
"$dst/bin/python" - <<"PY"
import importlib.metadata

assert importlib.metadata.version("nvidia-nccl-cu13") == "2.28.9"
assert importlib.metadata.version("nccl4py") == "0.2.0"
assert importlib.metadata.version("vllm") == "0.25.1"
PY
'
    printf 'prepared_at_utc=%s\nsource_commit=%s\nvllm_version=0.25.1\nnccl_version=2.28.9\n' \
        "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" >"$vllm_local_build/VENV_SUCCESS"
    tar -C "$vllm_local_build" -cf "$vllm_archive_build" .
    mv "$vllm_archive_build" "$vllm_archive"
    printf 'prepared_at_utc=%s\nsource_commit=%s\nvllm_version=0.25.1\nnccl_version=2.28.9\narchive=%s\n' \
        "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" "$vllm_archive" >"${vllm_archive_marker}.tmp"
    mv "${vllm_archive_marker}.tmp" "$vllm_archive_marker"
fi
[[ -s $vllm_archive && -s $vllm_archive_marker ]] || exit 2

if [[ ! -s $gym_marker ]]; then
    gym_archive_build="${gym_archive}.build.${job_id}"
    [[ ! -e $gym_archive ]] || { echo "Gym archive exists without marker: $gym_archive" >&2; exit 2; }
    [[ ! -e $gym_archive_build ]] || { echo "stale Gym archive build exists: $gym_archive_build" >&2; exit 2; }
    "${container[@]}" /bin/bash -ce '
trap "ray stop --force >/dev/null 2>&1 || true" EXIT
mkdir -p /job-tmp/uv /job-tmp/hf
PERSISTENT_CACHE=/opt UV_CACHE_DIR=/job-tmp/uv HF_HOME=/job-tmp/hf \
  python examples/nemo_gym/prefetch_venvs.py "$1"
' -- "$TARGET_CONFIG"
    find "$gym_local_build" -mindepth 1 -maxdepth 3 -type d -name .venv -print -quit | grep -q . || {
        echo "NeMo-Gym prefetch produced no cached venv" >&2
        exit 2
    }
    tar -C "$gym_local_build" -cf "$gym_archive_build" .
    mv "$gym_archive_build" "$gym_archive"
    printf 'prepared_at_utc=%s\nsource_commit=%s\nconfig=%s\narchive=%s\n' \
        "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" "$TARGET_CONFIG" "$gym_archive" \
        >"${gym_marker}.tmp"
    mv "${gym_marker}.tmp" "$gym_marker"
fi
[[ -s $gym_archive && -s $gym_marker ]] || exit 2

{
    printf 'prepared_at_utc=%s\njob_id=%s\nsource_commit=%s\nsource_dirty=%s\n' \
        "$(date -u +%FT%TZ)" "$job_id" "$SOURCE_COMMIT" "$SOURCE_DIRTY"
    printf 'container_image=%s\nmodel=%s\ntrain_data=%s\nvalidation_data=%s\n' \
        "$CONTAINER_IMAGE" "$MODEL_PATH" "$TRAIN_PATH" "$VAL_PATH"
    printf 'gym_venv_archive=%s\nvllm_worker_archive=%s\n' "$gym_archive" "$vllm_archive"
} >"${prep_marker}.tmp"
mv "${prep_marker}.tmp" "$prep_marker"
echo "Preparation complete: $prep_marker"
