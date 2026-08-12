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
# shellcheck source=/dev/null
source "$run_dir/control/run.env"

readonly image_uri="docker://nvcr.io/nvidia/nemo-rl:v0.7.0"
readonly dataset_id="nvidia/Nemotron-3-Nano-RL-Training-Blend"
readonly prep_marker="$run_dir/status/PREP_SUCCESS"
readonly failure_marker="$run_dir/status/PREP_FAILED"
readonly vllm_actor="nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
readonly vllm_venv_root="$CACHE_ROOT/ray_venvs/$SOURCE_COMMIT"
readonly vllm_venv_dir="$vllm_venv_root/$vllm_actor"
readonly vllm_venv_marker="$vllm_venv_dir/VENV_SUCCESS"
readonly raw_data_dir="$CACHE_ROOT/huggingface/nanov3_data/raw"
readonly prepared_data_dir="$CACHE_ROOT/huggingface/nanov3_data/prepared"
readonly data_marker="$prepared_data_dir/DATA_SUCCESS"
readonly gym_patch_rel="examples/bluevela/grpo_nanov3_30ba3b_32n8g/nemo_gym_multinic_vllm.patch"
readonly gym_submodule_rel="3rdparty/Gym-workspace/Gym"
job_id=${LSB_JOBID:-$$}
node_tmp="/tmp/nrl-nanov3-prep-${job_id}"

case "$node_tmp" in
    /tmp/nrl-nanov3-prep-[0-9]*) ;;
    *)
        echo "refusing unsafe temporary path: ${node_tmp}" >&2
        exit 2
        ;;
esac

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
mkdir -p \
    "$node_tmp/apptainer-tmp" "$node_tmp/ray" \
    "$run_dir/logs" "$run_dir/status" "$run_dir/tmp" "$run_dir/home" \
    "$CACHE_ROOT/apptainer-cache" "$CACHE_ROOT/huggingface/gym_venvs" \
    "$CACHE_ROOT/huggingface" "$CACHE_ROOT/hf_modules" \
    "$CACHE_ROOT/uv" "$CACHE_ROOT/xdg" "$CACHE_ROOT/torch" \
    "$CACHE_ROOT/triton" "$CACHE_ROOT/inductor" \
    "$vllm_venv_root" "$raw_data_dir" \
    "$CACHE_ROOT/vllm_compile_cache" "$CACHE_ROOT/flashinfer_cubins" \
    "$CACHE_ROOT/flashinfer_workspace" "$(dirname -- "$CONTAINER_IMAGE")"

printf 'started_at_utc=%s\njob_id=%s\nhost=%s\n' \
    "$(date -u +%FT%TZ)" "$job_id" "$(hostname -f)" \
    >"$run_dir/logs/prep-status.log"

[[ -x $APPTAINER ]] || {
    echo "Apptainer is not executable: ${APPTAINER}" >&2
    exit 2
}

source_dir="$run_dir/source"
if [[ ! -d $source_dir/.git ]]; then
    [[ ! -e $source_dir ]] || {
        echo "source path exists but is not a Git checkout: ${source_dir}" >&2
        exit 2
    }
    mkdir -p "$source_dir"
    git -C "$source_dir" init
    git -C "$source_dir" remote add origin "$SOURCE_REPO"
    git -C "$source_dir" fetch --depth 1 origin "$SOURCE_COMMIT"
    git -C "$source_dir" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$SOURCE_COMMIT" ]] || {
    echo "source checkout is not pinned to ${SOURCE_COMMIT}" >&2
    exit 2
}
git -C "$source_dir" submodule sync --recursive
git -C "$source_dir" submodule update --init --recursive --depth 1
if git -C "$source_dir" submodule status --recursive | grep -Eq '^[+-U]'; then
    echo "one or more submodules are missing or not pinned" >&2
    git -C "$source_dir" submodule status --recursive >&2
    exit 2
fi
gym_patch="$source_dir/$gym_patch_rel"
gym_dir="$source_dir/$gym_submodule_rel"
[[ -f $gym_patch && -d $gym_dir ]] || {
    echo "required NeMo-Gym multi-NIC patch or submodule is missing" >&2
    exit 2
}
if git -C "$gym_dir" apply --unidiff-zero --reverse --check "$gym_patch"; then
    echo "NeMo-Gym multi-NIC patch is already applied"
elif git -C "$gym_dir" apply --unidiff-zero --check "$gym_patch"; then
    git -C "$gym_dir" apply --unidiff-zero "$gym_patch"
else
    echo "NeMo-Gym multi-NIC patch does not apply cleanly" >&2
    exit 2
fi
[[ -f "$source_dir/$TARGET_ENTRYPOINT" ]] || {
    echo "entrypoint is absent at pinned commit: ${TARGET_ENTRYPOINT}" >&2
    exit 2
}
[[ -f "$source_dir/$TARGET_CONFIG" ]] || {
    echo "config is absent at pinned commit: ${TARGET_CONFIG}" >&2
    exit 2
}

validate_image() {
    "$APPTAINER" exec --fakeroot --containall --no-mount home --pwd / \
        "$CONTAINER_IMAGE" /bin/bash -ce \
        'test -d /opt/nemo-rl; test -x /opt/nemo_rl_venv/bin/python; command -v uv >/dev/null; command -v hf >/dev/null'
}

if [[ -s $CONTAINER_IMAGE ]]; then
    validate_image || {
        echo "existing image is not a compatible NeMo-RL v0.7.0 image: ${CONTAINER_IMAGE}" >&2
        exit 2
    }
else
    local_sif="$node_tmp/nemo-rl-v0.7.0.sif"
    partial_sif="${CONTAINER_IMAGE}.partial.${job_id}"
    [[ ! -e $partial_sif ]] || {
        echo "refusing existing partial image: ${partial_sif}" >&2
        exit 2
    }
    APPTAINER_TMPDIR="$node_tmp/apptainer-tmp" \
    APPTAINER_CACHEDIR="$CACHE_ROOT/apptainer-cache" \
        "$APPTAINER" build --fakeroot "$local_sif" "$image_uri"
    [[ -s $local_sif ]] || {
        echo "Apptainer build did not produce a SIF" >&2
        exit 2
    }
    cp "$local_sif" "$partial_sif"
    chmod 444 "$partial_sif"
    mv "$partial_sif" "$CONTAINER_IMAGE"
    validate_image
fi
sha256sum "$CONTAINER_IMAGE" >"$run_dir/status/container.sha256"

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
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

container=(
    "$APPTAINER" exec
    --fakeroot
    --containall
    --writable-tmpfs
    --no-mount home
    --bind "$source_dir:/opt/nemo-rl"
    --bind "$CACHE_ROOT:$CACHE_ROOT"
    --bind "$run_dir:$run_dir"
    --bind "$node_tmp:/job-tmp"
    --bind "$CACHE_ROOT/huggingface/gym_venvs:/opt/gym_venvs"
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
command -v uv
command -v hf
python -c "import nemo_rl, ray, torch; print(nemo_rl.__version__, ray.__version__, torch.__version__)"
'

"${container[@]}" hf download "$MODEL_ID"
model_cache_root="$CACHE_ROOT/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/snapshots"
model_snapshot=$(find "$model_cache_root" -mindepth 2 -maxdepth 2 \
    -name config.json -printf '%h\n' | sort | tail -n 1)
[[ -n $model_snapshot && -s $model_snapshot/config.json ]] || {
    echo "Nano v3 model snapshot validation failed: ${model_snapshot}" >&2
    exit 2
}
printf 'model=%s\nsnapshot=%s\n' "$MODEL_ID" "$model_snapshot" \
    >"$run_dir/status/model.ready"

"${container[@]}" hf download "$TOKENIZER_ID" \
    tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja
tokenizer_cache_root="$CACHE_ROOT/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots"
tokenizer_snapshot=$(find "$tokenizer_cache_root" -mindepth 2 -maxdepth 2 \
    -name tokenizer_config.json -printf '%h\n' | sort | tail -n 1)
[[ -n $tokenizer_snapshot && -s $tokenizer_snapshot/tokenizer_config.json ]] || {
    echo "Nano v3 tokenizer snapshot validation failed: ${tokenizer_snapshot}" >&2
    exit 2
}
"${container[@]}" python - "$tokenizer_snapshot/tokenizer_config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as config_file:
    config = json.load(config_file)
assert config.get("chat_template"), "Nano v3 tokenizer is missing chat_template"
PY
printf 'tokenizer=%s\nsnapshot=%s\n' "$TOKENIZER_ID" "$tokenizer_snapshot" \
    >"$run_dir/status/tokenizer.ready"

if [[ ! -s $data_marker ]]; then
    "${container[@]}" hf download "$dataset_id" \
        --repo-type dataset --local-dir "$raw_data_dir"
    [[ -s "$raw_data_dir/train.jsonl" ]] || {
        echo "downloaded Nano v3 training data is missing train.jsonl" >&2
        exit 2
    }
    [[ -s "$raw_data_dir/create_nanov3_jsonl.py" ]] || {
        echo "downloaded Nano v3 dataset is missing create_nanov3_jsonl.py" >&2
        exit 2
    }

    data_build="${prepared_data_dir}.build.${job_id}"
    [[ ! -e $data_build ]] || {
        echo "refusing existing data build path: ${data_build}" >&2
        exit 2
    }
    mkdir -p "$data_build"
    "${container[@]}" python "$raw_data_dir/create_nanov3_jsonl.py" \
        --input "$raw_data_dir/train.jsonl" \
        --output "$data_build/train-full.jsonl"
    total_rows=$(wc -l <"$data_build/train-full.jsonl")
    (( total_rows > 1000 )) || {
        echo "prepared Nano v3 dataset has only ${total_rows} rows" >&2
        exit 2
    }
    train_rows=$((total_rows - 1000))
    head -n "$train_rows" "$data_build/train-full.jsonl" \
        >"$data_build/train-split.jsonl"
    tail -n 1000 "$data_build/train-full.jsonl" \
        >"$data_build/val-split.jsonl"
    [[ -s "$data_build/train-split.jsonl" && -s "$data_build/val-split.jsonl" ]] || {
        echo "Nano v3 train/validation split is empty" >&2
        exit 2
    }
    printf 'prepared_at_utc=%s\nsource=%s\ntotal_rows=%s\ntrain_rows=%s\nvalidation_rows=1000\n' \
        "$(date -u +%FT%TZ)" "$dataset_id" "$total_rows" "$train_rows" \
        >"$data_build/DATA_SUCCESS"
    [[ ! -e $prepared_data_dir ]] || {
        echo "prepared data path exists without a success marker: ${prepared_data_dir}" >&2
        exit 2
    }
    mv "$data_build" "$prepared_data_dir"
fi
[[ -s "$prepared_data_dir/train-split.jsonl" ]]
[[ -s "$prepared_data_dir/val-split.jsonl" ]]

if [[ ! -s $vllm_venv_marker ]]; then
    vllm_venv_build="${vllm_venv_dir}.build.${job_id}"
    [[ ! -e $vllm_venv_build ]] || {
        echo "refusing existing vLLM venv build path: ${vllm_venv_build}" >&2
        exit 2
    }
    mkdir -p "$vllm_venv_build"
    # shellcheck disable=SC2016
    "${container[@]}" /bin/bash -ce '
unset UV_NO_SYNC UV_PROJECT_ENVIRONMENT VIRTUAL_ENV
build_dir=$1
actor=$2
cp -a --reflink=auto "/opt/ray_venvs/$actor/." "$build_dir/"
ray_version=$("$build_dir/bin/python" -c "import importlib.metadata; print(importlib.metadata.version(\"ray\"))")
torch_version=$("$build_dir/bin/python" -c "import importlib.metadata; print(importlib.metadata.version(\"torch\"))")
uv pip install \
  --python "$build_dir/bin/python" \
  --index https://flashinfer.ai/whl/cu130 \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "flashinfer-jit-cache==0.6.13+cu130"
PYTHONPATH=/opt/nemo-rl "$build_dir/bin/python" - "$ray_version" "$torch_version" <<"PY"
import importlib.metadata
import sys

expected_ray, expected_torch = sys.argv[1:]
assert importlib.metadata.version("vllm") == "0.25.1"
assert importlib.metadata.version("flashinfer-jit-cache") == "0.6.13+cu130"
assert importlib.metadata.version("ray") == expected_ray
assert importlib.metadata.version("torch") == expected_torch
PY
' -- "$vllm_venv_build" "$vllm_actor"
    printf 'prepared_at_utc=%s\nsource_commit=%s\nvllm_version=0.25.1\n' \
        "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" >"$vllm_venv_build/VENV_SUCCESS"
    [[ ! -e $vllm_venv_dir ]] || {
        echo "vLLM venv path exists without a success marker: ${vllm_venv_dir}" >&2
        exit 2
    }
    mv "$vllm_venv_build" "$vllm_venv_dir"
fi
[[ -L "$vllm_venv_dir/bin/python" && -s $vllm_venv_marker ]]

gym_venv_dir="$CACHE_ROOT/huggingface/gym_venvs"
gym_marker="$gym_venv_dir/GYM_SUCCESS"
if [[ ! -s $gym_marker ]]; then
    # Positional parameters are expanded by the container-side shell.
    # shellcheck disable=SC2016
    "${container[@]}" /bin/bash -ce '
trap "ray stop --force >/dev/null 2>&1 || true" EXIT
python examples/nemo_gym/prefetch_venvs.py "$1"
' -- "$TARGET_CONFIG"
    find "$gym_venv_dir" -mindepth 1 -maxdepth 3 -type d -name .venv -print -quit \
        | grep -q . || {
        echo "NeMo-Gym prefetch produced no cached venv" >&2
        exit 2
    }
    printf 'prepared_at_utc=%s\nsource_commit=%s\nconfig=%s\n' \
        "$(date -u +%FT%TZ)" "$SOURCE_COMMIT" "$TARGET_CONFIG" >"$gym_marker"
fi
[[ -s $gym_marker ]]

git -C "$source_dir" diff --quiet -- . ":(exclude)$gym_submodule_rel"
git -C "$gym_dir" diff -U0 -- responses_api_models/local_vllm_model/local_vllm_model_actor.py \
    | cmp - "$gym_patch"
git -C "$source_dir" diff --cached --quiet

{
    printf 'prepared_at_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'job_id=%s\n' "$job_id"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'container_image=%s\n' "$CONTAINER_IMAGE"
    printf 'container_sha256=%s\n' "$(cut -d ' ' -f 1 "$run_dir/status/container.sha256")"
    printf 'gym_patch_sha256=%s\n' "$(sha256sum "$gym_patch" | awk '{ print $1 }')"
    printf 'model_snapshot=%s\n' "$model_snapshot"
    printf 'train_data=%s\n' "$prepared_data_dir/train-split.jsonl"
    printf 'validation_data=%s\n' "$prepared_data_dir/val-split.jsonl"
    printf 'gym_venv_dir=%s\n' "$gym_venv_dir"
    printf 'vllm_worker_venv=%s\n' "$vllm_venv_dir"
} >"${prep_marker}.tmp"
mv "${prep_marker}.tmp" "$prep_marker"

echo "Preparation complete: ${prep_marker}"
