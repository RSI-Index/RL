"""Contract tests for the 32-node Blue Vela Nano v3 GRPO launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples/bluevela/grpo_nanov3_30ba3b_32n8g"
SUBMIT = EXAMPLE_DIR / "submit.sh"
CONFIG = "examples/nemo_gym/grpo_nanov3.yaml"
ENTRYPOINT = "examples/nemo_gym/run_grpo_nemo_gym.py"


def test_dry_run_renders_32_node_contract_without_writing(tmp_path: Path) -> None:
    run_dir = tmp_path / "must-not-exist"
    env = {
        **os.environ,
        "RUN_ID": "20260807T000000Z-nanov3-32n-yuetai",
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
    assert "-n 256" in rendered
    assert "span[ptile=8]" in rendered
    assert "num=8:mode=shared:j_exclusive=yes" in rendered
    assert "32 hosts x 8 GPUs" in rendered
    assert "done(PREP_JOB_ID)" in rendered
    assert CONFIG in rendered
    assert ENTRYPOINT in rendered
    assert not run_dir.exists()


def test_payloads_cover_model_data_gym_ray_and_markers() -> None:
    submit = (EXAMPLE_DIR / "submit.sh").read_text()
    prepare = (EXAMPLE_DIR / "prepare.sh").read_text()
    ray_node = (EXAMPLE_DIR / "ray_node.sh").read_text()
    train = (EXAMPLE_DIR / "train.sh").read_text()

    for payload in (submit, prepare, ray_node, train):
        assert "set -euo pipefail" in payload

    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" in submit
    assert 'hf download "$MODEL_ID"' in prepare
    assert "nvidia/Nemotron-3-Nano-RL-Training-Blend" in prepare
    assert "create_nanov3_jsonl.py" in prepare
    assert "train-split.jsonl" in prepare
    assert "val-split.jsonl" in prepare
    assert "prefetch_venvs.py" in prepare
    assert 'prefetch_venvs.py "$1"' in prepare
    assert '"$TARGET_CONFIG"' in prepare
    assert "vllm-0.25.1-cp38-abi3-manylinux_2_28_x86_64.whl" in prepare
    assert "PREP_SUCCESS" in prepare

    assert "ray start --head" in ray_node
    assert "ray start --address" in ray_node
    assert "--num-gpus=8" in ray_node
    assert "--min-worker-port=2000" in ray_node
    assert "--max-worker-port=2999" in ray_node
    assert "RAY_STOP" in ray_node

    assert "LSB_MCPU_HOSTS" in train
    assert "expected 32 allocated hosts" in train
    assert "blaunch" in train
    assert "ray_node.sh" in train
    assert "expected_gpu_total=256" in train
    assert ENTRYPOINT in train
    assert CONFIG in train
    assert "CHECKPOINT_DIR" in submit
    assert "CHECKPOINT_DIR" in train
    assert "data.train.data_path" in train
    assert "data.validation.data_path" in train
    assert "status/SUCCESS" in train
    assert "TRAIN_FAILED" in train
