"""Contract tests for the Blue Vela Super SWE1 example launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples/bluevela/grpo_qwen3_1_7b_super_swe1"
SUBMIT = EXAMPLE_DIR / "submit.sh"
TARGET = "tests/test_suites/llm/grpo-qwen3-1.7b-1n8g-megatron-super-swe1.sh"


def test_dry_run_renders_bluevela_contract_without_writing(tmp_path: Path) -> None:
    run_dir = tmp_path / "must-not-exist"
    env = {
        **os.environ,
        "RUN_ID": "20260807T000000Z-contract-yuetai",
        "RUN_DIR": str(run_dir),
        "SOURCE_COMMIT": "e496258b00000000000000000000000000000000",
    }

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    rendered = result.stdout
    assert "-q normal" in rendered
    assert "-G grp_models" in rendered
    assert "-n 64" in rendered
    assert "-M 524288" in rendered
    assert "select[tmp>153600]" in rendered
    assert "rusage[mem=524288]" in rendered
    assert "-n 8" in rendered
    assert "num=8:mode=shared:j_exclusive=yes" in rendered
    assert "mode=exclusive_process" not in rendered
    assert "gmodel=" not in rendered
    assert "done(PREP_JOB_ID)" in rendered
    assert TARGET in rendered
    assert not run_dir.exists()


def test_payloads_have_strict_markers_and_exact_target() -> None:
    prepare = (EXAMPLE_DIR / "prepare.sh").read_text()
    train = (EXAMPLE_DIR / "train.sh").read_text()

    assert "set -euo pipefail" in prepare
    assert "set -euo pipefail" in train
    assert "PREP_SUCCESS" in prepare
    assert "--fakeroot" in prepare
    assert "--fakeroot" in train
    assert "docker://nvcr.io/nvidia/nemo-rl:v0.7.0" in prepare
    assert "--nv" in train
    assert TARGET in train
    assert "status/SUCCESS" in train
    assert "/bin/bash -ceu" not in prepare
    assert "/bin/bash -ceu" not in train
    assert "/root/.local/bin" in prepare
    assert "/root/.local/bin" in train
    assert "models--Qwen--Qwen3-1.7B/snapshots" in prepare
    assert 'gym_venv_dir="$CACHE_ROOT/huggingface/gym_venvs"' in prepare
    assert '"$CACHE_ROOT/huggingface/gym_venvs/GYM_SUCCESS"' in train
    assert '"$CACHE_ROOT/huggingface/gym_venvs:/opt/gym_venvs"' in prepare
    assert '"$CACHE_ROOT/huggingface/gym_venvs:/opt/gym_venvs"' in train
    assert "uv export --directory /opt/nemo-rl" not in prepare
    assert "uv pip install" in prepare
    assert "uv pip install --no-deps" not in prepare
    assert "--index https://flashinfer.ai/whl/cu130" in prepare
    assert "vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl" in prepare
    assert '"flashinfer-jit-cache==0.6.13+cu130"' in prepare
    assert 'importlib.metadata.version("ray") == expected_ray' in prepare
    assert 'importlib.metadata.version("torch") == expected_torch' in prepare
    assert '"class ServingTokenization" in serving_source' in prepare
    assert "from vllm.entrypoints.serve.tokenize.serving import ServingTokenization" not in prepare
    assert 'cp -a --reflink=auto "/opt/ray_venvs/$actor/."' in prepare
    assert 'importlib.metadata.version("vllm") == "0.25.1"' in prepare
    assert '[[ -L "$vllm_venv_dir/bin/python" && -s $vllm_venv_marker ]]' in prepare
    assert '[[ -x "$vllm_venv_dir/bin/python"' not in prepare
    assert '[[ -s "$vllm_venv_dir/VENV_SUCCESS" && -L "$vllm_venv_dir/bin/python" ]]' in train
    assert '-x "$vllm_venv_dir/bin/python"' not in train
    assert '--env UV_PYTHON=/opt/nemo_rl_venv/bin/python' in train
    assert '--bind "$vllm_venv_dir:/opt/ray_venvs/$vllm_actor"' in train
    assert '"$CACHE_ROOT/gym_venvs/GYM_SUCCESS"' not in prepare
    assert '"$CACHE_ROOT/gym_venvs/GYM_SUCCESS"' not in train
    assert "v0.5" not in prepare
    assert "v0.5" not in train
