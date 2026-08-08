# Blue Vela Qwen3-1.7B Super SWE1

This example runs the existing five-step Super SWE1 nightly on one Blue Vela
node with eight H100 GPUs. It follows the Megatron baseline convention: code is
kept in the repository, while immutable run control files, logs, caches, the
v0.7.0 SIF, data, and model weights live under `/proj`.

The launcher defaults to a no-write dry run:

```bash
bash examples/bluevela/grpo_qwen3_1_7b_super_swe1/submit.sh --dry-run
```

Submit preparation and its dependent training job:

```bash
bash examples/bluevela/grpo_qwen3_1_7b_super_swe1/submit.sh --submit
```

Preparation uses queue `normal`, group `grp_models`, 64 CPU slots, 512 GiB,
one host, node-local `/tmp`, and a six-hour limit. Training is submitted at the
same time with `done(<prep-job-id>)` and uses the baseline
`num=8:mode=shared:j_exclusive=yes` request on one host. Job exclusivity keeps
other jobs off the allocated GPUs, while shared compute mode lets the colocated
vLLM and Megatron worker processes use each GPU. It validates all eight GPUs as
H100s at runtime and requests 512 GiB for three hours. It runs this file without
modification:

```text
tests/test_suites/llm/grpo-qwen3-1.7b-1n8g-megatron-super-swe1.sh
```

The prep job also builds a commit-scoped vLLM 0.25.1 Ray-worker venv. The v0.7
SIF bundles vLLM 0.20.0, while the pinned source commit requires the newer API;
training overlays only that worker venv and leaves the rest of the SIF intact.

The submitter prints the unique run directory and both job IDs. Inspect them
with:

```bash
bjobs -l PREP_JOB_ID TRAIN_JOB_ID
tail -f RUN_DIR/logs/prep.PREP_JOB_ID.out
tail -f RUN_DIR/logs/train.TRAIN_JOB_ID.out
```

LSF `DONE` is not sufficient. Preparation is complete only when
`RUN_DIR/status/PREP_SUCCESS` exists. Training is complete only when
`RUN_DIR/status/SUCCESS` exists; that marker is written after the original
nightly exits successfully and `metrics.json` contains step 5.

The launcher reads the existing mode-600 credential files at
`~/.config/megatron-bridge/wandb_api_key` and `~/.cache/huggingface/token`.
Their values are never copied into the run directory or printed.
