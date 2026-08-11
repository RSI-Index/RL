"""Contract tests for the Blue Vela Super 120B 27-node launcher."""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples/bluevela/grpo_super_120b_27n8g"


def test_vllm_archive_pins_policy_compatible_nccl() -> None:
    prepare = (EXAMPLE_DIR / "prepare.sh").read_text()
    train = (EXAMPLE_DIR / "train.sh").read_text()
    ray_node = (EXAMPLE_DIR / "ray_node.sh").read_text()

    archive_tag = "vllm-0.25.1-nccl-2.28.9"
    assert "uv pip install --no-config --python" in prepare
    assert '"nvidia-nccl-cu13==2.28.9"' in prepare
    assert 'importlib.metadata.version("nvidia-nccl-cu13") == "2.28.9"' in prepare
    assert archive_tag in prepare
    assert archive_tag in train
    assert archive_tag in ray_node


def test_training_allocation_covers_observed_async_memory_peak() -> None:
    env = {
        **os.environ,
        "MODEL_PATH": "/proj/model",
        "TRAIN_PATH": "/proj/train.jsonl",
        "VAL_PATH": "/proj/val.jsonl",
    }
    result = subprocess.run(
        ["bash", str(EXAMPLE_DIR / "submit.sh"), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "-M 1048576" in result.stdout
    assert "rusage\\[mem=1048576\\]" in result.stdout
    assert "-W 24:00" in result.stdout
