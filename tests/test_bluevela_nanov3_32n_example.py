"""Contract tests for the 32-node Blue Vela Nano v3 GRPO launcher."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples/bluevela/grpo_nanov3_30ba3b_32n8g"
SUBMIT = EXAMPLE_DIR / "submit.sh"
CONFIG = "examples/nemo_gym/grpo_nanov3.yaml"
ENTRYPOINT = "examples/nemo_gym/run_grpo_nemo_gym.py"
TOKENIZER_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


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
    assert "-n 544" in rendered
    assert "span[ptile=17]" in rendered
    assert "num=8:mode=shared:j_exclusive=yes" in rendered
    assert "32 hosts x 8 GPUs" in rendered
    assert "done(PREP_JOB_ID)" in rendered
    assert CONFIG in rendered
    assert ENTRYPOINT in rendered
    assert not run_dir.exists()


def test_dry_run_renders_two_node_batch_contract_without_training_overrides(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "must-not-exist"
    env = {
        **os.environ,
        "RUN_ID": "20260808T040000Z-nanov3-2n-smoke-yuetai",
        "RUN_DIR": str(run_dir),
        "SOURCE_COMMIT": "e496258b00000000000000000000000000000000",
        "TRAIN_HOSTS": "2",
        "TRAIN_WALLTIME": "02:00",
    }

    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    rendered = result.stdout
    assert "-n 34" in rendered
    assert "2 hosts x 8 GPUs = 16 GPUs" in rendered
    assert "-W 02:00" in rendered
    assert "Smoke mode" not in rendered
    assert not run_dir.exists()


def test_payloads_cover_model_data_gym_ray_and_markers() -> None:
    submit = (EXAMPLE_DIR / "submit.sh").read_text()
    prepare = (EXAMPLE_DIR / "prepare.sh").read_text()
    ray_node = (EXAMPLE_DIR / "ray_node.sh").read_text()
    train = (EXAMPLE_DIR / "train.sh").read_text()

    for payload in (submit, prepare, ray_node, train):
        assert "set -euo pipefail" in payload

    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" in submit
    assert TOKENIZER_ID in submit
    assert 'hf download "$MODEL_ID"' in prepare
    assert 'hf download "$TOKENIZER_ID" \\' in prepare
    for tokenizer_file in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        assert tokenizer_file in prepare
    assert '"${container[@]}" hf download "$TOKENIZER_ID"\n' not in prepare
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
    assert "container_start_attempts=${CONTAINER_START_ATTEMPTS:-3}" in ray_node
    assert "status != 255 || attempt == container_start_attempts" in ray_node
    assert "Apptainer startup returned status 255" in ray_node
    for payload in (ray_node, train):
        assert "--bind /dev/infiniband:/dev/infiniband" in payload
        assert "RDMA devices" in payload
        assert "APPTAINERENV_NCCL_DEBUG=INFO" in payload
        assert "APPTAINERENV_NCCL_DEBUG_SUBSYS=INIT,NET" in payload

    assert "LSB_MCPU_HOSTS" in train
    assert 'expected ${expected_host_count} allocated hosts' in train
    assert '${expected_slots_per_host} slots each' in train
    assert "blaunch" in train
    assert "ray_node.sh" in train
    assert "expected_gpu_total=$((expected_host_count * 8))" in train
    assert ENTRYPOINT in train
    assert CONFIG in train
    assert "CHECKPOINT_DIR" in submit
    assert 'printf \'training_shape=%s hosts x 8 GPUs\\n\' "$train_hosts"' in submit
    assert "CHECKPOINT_DIR" in train
    assert "data.train.data_path" in train
    assert "data.validation.data_path" in train
    assert "status/SUCCESS" in train
    assert "TRAIN_FAILED" in train
    assert "smoke_overrides" not in train
    for forbidden_override in (
        "grpo.num_prompts_per_step=2",
        "grpo.num_generations_per_prompt=4",
        "grpo.max_num_steps=1",
        "policy.train_global_batch_size=4",
        "policy.generation.max_new_tokens=512",
        "checkpointing.enabled=false",
    ):
        assert forbidden_override not in train


def test_train_driver_shares_host_pid_namespace_with_ray_head(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    cache_root = tmp_path / "cache"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (run_dir / "control").mkdir(parents=True)
    (run_dir / "source").mkdir()
    (run_dir / "status/ray").mkdir(parents=True)
    (run_dir / "logs/ray").mkdir(parents=True)

    source_commit = "e496258b0285d42d5d9af30671e81722bca916dc"
    actor = "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
    prepared_data = cache_root / "huggingface/nanov3_data/prepared"
    gym_venv = cache_root / "huggingface/gym_venvs"
    actor_venv = cache_root / "ray_venvs" / source_commit / actor
    prepared_data.mkdir(parents=True)
    gym_venv.mkdir(parents=True)
    (actor_venv / "bin").mkdir(parents=True)
    (prepared_data / "train-split.jsonl").write_text("{}\n")
    (prepared_data / "val-split.jsonl").write_text("{}\n")
    (gym_venv / "GYM_SUCCESS").write_text("ok\n")
    (actor_venv / "VENV_SUCCESS").write_text("ok\n")
    (actor_venv / "bin/python").symlink_to("/opt/nemo_rl_venv/bin/python")
    (run_dir / "status/PREP_SUCCESS").write_text("ok\n")

    image = tmp_path / "nemo-rl.sif"
    image.write_text("fake image\n")
    wandb_key = tmp_path / "wandb-key"
    hf_token = tmp_path / "hf-token"
    wandb_key.write_text("not-a-real-secret\n")
    hf_token.write_text("not-a-real-secret\n")
    wandb_key.chmod(0o600)
    hf_token.chmod(0o600)

    apptainer_log = tmp_path / "apptainer-args.log"
    uv_log = tmp_path / "uv-args.log"
    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' CALL \"$@\" >>\"$APPTAINER_ARGS_LOG\"\n"
        "while [[ $# -gt 0 && $1 != \"$CONTAINER_IMAGE\" ]]; do shift; done\n"
        "[[ $# -gt 0 ]] || exit 1\n"
        "shift\n"
        "if [[ ${1:-} == /bin/bash ]]; then\n"
        r'  script=${3/cd \/opt\/nemo-rl/cd "$RUN_DIR\/source"}' "\n"
        "  shift 3\n"
        "  exec /bin/bash -ce \"$script\" \"$@\"\n"
        "fi\n"
    )
    fake_apptainer.chmod(0o700)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" >\"$UV_ARGS_LOG\"\n"
    )
    fake_uv.chmod(0o700)

    fake_git = fake_bin / "git"
    fake_git.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' {source_commit}\n")
    fake_git.chmod(0o700)

    fake_ip = fake_bin / "ip"
    fake_ip.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '1: ibp26s0    inet 100.126.0.1/24 scope global ibp26s0'\n"
    )
    fake_ip.chmod(0o700)

    fake_blaunch = fake_bin / "blaunch"
    fake_blaunch.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "run_dir=$5\n"
        "rank=$8\n"
        "mkdir -p \"$run_dir/status/ray\"\n"
        "printf 'ready\\n' >\"$run_dir/status/ray/node-${rank}.ready\"\n"
        "if [[ $rank == 0 ]]; then\n"
        "  mkdir -p \"/tmp/nrl-nanov3-ray-${LSB_JOBID}-0\"\n"
        "fi\n"
    )
    fake_blaunch.chmod(0o700)

    ray_node = run_dir / "control/ray_node.sh"
    ray_node.write_text("#!/usr/bin/env bash\nexit 0\n")
    ray_node.chmod(0o700)
    checkpoint_dir = run_dir / "checkpoints"
    (run_dir / "control/run.env").write_text(
        "\n".join(
            (
                "export RUN_ID=pid-namespace-test",
                f"export RUN_DIR={run_dir}",
                f"export CHECKPOINT_DIR={checkpoint_dir}",
                f"export CACHE_ROOT={cache_root}",
                f"export CONTAINER_IMAGE={image}",
                f"export APPTAINER={fake_apptainer}",
                f"export SOURCE_COMMIT={source_commit}",
                f"export WANDB_API_KEY_FILE={wandb_key}",
                f"export HF_TOKEN_FILE={hf_token}",
                f"export TARGET_ENTRYPOINT={ENTRYPOINT}",
                f"export TARGET_CONFIG={CONFIG}",
                "export MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
                f"export TOKENIZER_ID={TOKENIZER_ID}",
                "export TRAIN_EXPECTED_HOSTS=2",
                "export RAY_EXPECTED_HOSTS=2",
                "",
            )
        )
    )

    job_id = str(os.getpid())
    head_tmp = Path(f"/tmp/nrl-nanov3-ray-{job_id}-0")
    env = {
        **os.environ,
        "APPTAINER_ARGS_LOG": str(apptainer_log),
        "UV_ARGS_LOG": str(uv_log),
        "LSB_JOBID": job_id,
        "LSB_MCPU_HOSTS": f"{socket.gethostname().split('.')[0]} 8 worker 8",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    try:
        result = subprocess.run(
            ["bash", "-x", str(EXAMPLE_DIR / "train.sh"), str(run_dir)],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )
    finally:
        shutil.rmtree(head_tmp, ignore_errors=True)

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    apptainer_args = apptainer_log.read_text().splitlines()
    assert "--contain" in apptainer_args
    assert "--containall" not in apptainer_args
    assert "cluster.num_nodes=2" in uv_log.read_text().splitlines()
    assert f"policy.tokenizer.name={TOKENIZER_ID}" in uv_log.read_text().splitlines()
    assert (
        f"+policy.generation.vllm_kwargs.tokenizer={TOKENIZER_ID}"
        in uv_log.read_text().splitlines()
    )
    forbidden_training_overrides = {
        "grpo.num_prompts_per_step=2",
        "grpo.num_generations_per_prompt=4",
        "grpo.max_num_steps=1",
        "policy.train_global_batch_size=4",
        "policy.generation.max_new_tokens=512",
        "checkpointing.enabled=false",
    }
    assert forbidden_training_overrides.isdisjoint(uv_log.read_text().splitlines())
    assert (run_dir / "status/SUCCESS").is_file()


def test_ray_node_advertises_cpus_for_colocated_policy_and_generation(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    cache_root = tmp_path / "cache"
    fake_bin = tmp_path / "bin"
    (run_dir / "control").mkdir(parents=True)
    (run_dir / "status").mkdir()
    fake_bin.mkdir()

    source_commit = "e496258b0285d42d5d9af30671e81722bca916dc"
    actor = "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
    actor_venv = cache_root / "ray_venvs" / source_commit / actor
    (actor_venv / "bin").mkdir(parents=True)
    (actor_venv / "VENV_SUCCESS").write_text("ok\n")
    (actor_venv / "bin/python").symlink_to("/opt/nemo_rl_venv/bin/python")
    (run_dir / "status/PREP_SUCCESS").write_text("ok\n")
    (run_dir / "status/RAY_STOP").write_text("stop\n")

    image = tmp_path / "nemo-rl.sif"
    image.write_text("fake image\n")
    wandb_key = tmp_path / "wandb-key"
    hf_token = tmp_path / "hf-token"
    wandb_key.write_text("not-a-real-secret\n")
    hf_token.write_text("not-a-real-secret\n")
    wandb_key.chmod(0o600)
    hf_token.chmod(0o600)

    ray_log = tmp_path / "ray-args.log"
    fake_ray = fake_bin / "ray"
    fake_ray.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' CALL \"$@\" >>\"$RAY_ARGS_LOG\"\n"
    )
    fake_ray.chmod(0o700)

    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "for _ in {1..8}; do printf '%s\\n' 'NVIDIA H100 80GB HBM3'; done\n"
    )
    fake_nvidia_smi.chmod(0o700)

    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text(
        "#!/usr/bin/env bash\n"
        "while [[ $# -gt 0 && $1 != \"$CONTAINER_IMAGE\" ]]; do shift; done\n"
        "[[ $# -gt 0 ]] || exit 1\n"
        "shift\n"
        "exec \"$@\"\n"
    )
    fake_apptainer.chmod(0o700)

    (run_dir / "control/run.env").write_text(
        "\n".join(
            (
                f"export CACHE_ROOT={cache_root}",
                f"export CONTAINER_IMAGE={image}",
                f"export APPTAINER={fake_apptainer}",
                f"export SOURCE_COMMIT={source_commit}",
                f"export WANDB_API_KEY_FILE={wandb_key}",
                f"export HF_TOKEN_FILE={hf_token}",
                "export RAY_EXPECTED_HOSTS=1",
                "",
            )
        )
    )

    result = subprocess.run(
        [
            "bash",
            str(EXAMPLE_DIR / "ray_node.sh"),
            str(run_dir),
            "head",
            "100.126.0.1:1200",
            "0",
        ],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "LSB_JOBID": f"{os.getpid()}7",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "RAY_ARGS_LOG": str(ray_log),
        },
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "--num-cpus=17" in ray_log.read_text().splitlines()
