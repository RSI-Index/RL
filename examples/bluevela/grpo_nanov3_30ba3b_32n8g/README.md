# Blue Vela Nemotron-3 Nano 32-node GRPO

This launcher runs the published Nano v3 NeMo-Gym GRPO configuration on 32
Blue Vela hosts with eight H100 GPUs per host. It executes these repository
files without changing their training hyperparameters:

```text
examples/nemo_gym/run_grpo_nemo_gym.py
examples/bluevela/grpo_nanov3_30ba3b_32n8g/config.yaml
```

The directory also carries `nemo_gym_multinic_vllm.patch`. Preparation applies
and verifies this pinned patch against the NeMo-Gym submodule so vLLM's Ray
data-parallel workers use the same network interface that Ray advertises on
multi-NIC Blue Vela hosts.

The only Hydra overrides are operational: the prepared model and data paths,
run-specific log/checkpoint paths, and W&B enablement/name.

The launcher defaults to a no-write dry run:

```bash
bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --dry-run
```

By default, preparation fetches the exact local commit from
`https://github.com/RSI-Index/RL.git`; therefore submit only commits that have
already been pushed to that repository. `SOURCE_REPO` remains available for
testing an explicitly chosen mirror.

Review the rendered 32-host request carefully. Submit the run when ready:

```bash
bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --submit
```

When SForge already holds a native interactive LSF allocation, reuse those
nodes without submitting child jobs:

```bash
bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --within-allocation
```

This mode requires `LSB_JOBID` and `LSB_MCPU_HOSTS`, runs preparation when no
validated cache can be reused, and then runs training in the current allocation.

To avoid hosts with a confirmed fabric fault on a retry, pass a validated,
space-separated exclusion list through `TRAIN_EXCLUDE_HOSTS`. The exclusion is
applied only to LSF host selection and does not change the training config:

```bash
TRAIN_EXCLUDE_HOSTS="host-a host-b" \
  bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --submit
```

Before submitting a preparation job, the launcher searches previous runs for a
successful preparation from the exact same source commit. It validates the
source checkout, container hash, model and tokenizer snapshots, prepared data,
Gym environments, and commit-scoped vLLM worker environment. When every check
passes, it reflink-copies the immutable source checkout into the new run and
submits training directly without an LSF preparation dependency.

When no reusable preparation passes validation, preparation downloads and
validates the v0.7.0 SIF, a commit-pinned source
checkout, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`, and
`nvidia/Nemotron-3-Nano-RL-Training-Blend`. It fills dataset placeholders,
uses the final 1,000 rows for validation, builds a commit-scoped vLLM 0.25.1
Ray-worker environment, and prefetches the NeMo-Gym environments.

Training requests queue `priority`, group `grp_models`, 2,048 slots spread 64
per host, 512 GiB per host, `num=8:mode=shared:j_exclusive=yes`, and LSF `-x`.
The GPU requirement prevents cross-job GPU sharing, while `-x` reserves every
allocated host exclusively so CPU-only jobs cannot be colocated with training.
The training payload validates exactly 32 allocated hosts and eight H100s on
every host before starting one Ray head plus 31 workers through `blaunch`.

The submitter prints the run directory and both LSF job IDs. Monitor with:

```bash
bjobs -l PREP_JOB_ID TRAIN_JOB_ID
tail -f RUN_DIR/logs/prep.PREP_JOB_ID.out
tail -f RUN_DIR/logs/train.TRAIN_JOB_ID.out
tail -f RUN_DIR/logs/train-driver.log
```

Preparation is complete only when `RUN_DIR/status/PREP_SUCCESS` exists.
Training is complete only when `RUN_DIR/status/SUCCESS` exists. Failures write
`PREP_FAILED`, `TRAIN_FAILED`, or per-node Ray logs under `RUN_DIR/logs/ray/`.

The recipe's `checkpoint_must_save_by: 00:03:40:00` intentionally checkpoints
and exits before the four-hour LSF limit.

To override only the step-based checkpoint cadence for a future allocation,
set `CHECKPOINT_SAVE_PERIOD` when submitting. For example, the following saves
every five training steps while leaving the time-based deadline unchanged:

```bash
CHECKPOINT_SAVE_PERIOD=5 \
  bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --submit
```

If the epoch needs another allocation, submit a new immutable run directory
while pointing it at the prior checkpoint directory:

```bash
CHECKPOINT_DIR=/proj/.../previous-run/checkpoints \
  bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --submit
```

NeMo-RL discovers the latest `step_*` checkpoint in that directory and restores
the model, optimizer, dataloader, epoch, and step state.

Credentials are read from mode-600 files at
`~/.config/megatron-bridge/wandb_api_key` and
`~/.cache/huggingface/token`. Secret values are not copied into the run
directory or printed.
