# Blue Vela Nemotron 3 Super 27-node RLVR

This launcher follows the Blue Vela layout used by
`grpo_nanov3_30ba3b_32n8g`, while targeting the current local Super RLVR
recipe:

```text
examples/nemo_gym/run_grpo_nemo_gym.py
examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
```

The allocation is 27 H100 hosts with eight GPUs per host:

```text
20 NeMo-RL hosts = 8 training hosts + 12 generation hosts
7 additional hosts = NeMo-Gym/judge capacity
```

The launcher requests 17 LSF slots and 1 TiB per host by default. Super 120B
async training overlaps the driver training batch with replay-buffer rollout
payloads and has exceeded a 512 GiB LSF memory limit in practice. Values below
1 TiB are rejected; a larger value can be selected with `TRAIN_MEMORY_MB`.
The default walltime is 24 hours and can be changed with `TRAIN_WALLTIME`. GPU
allocation is job-exclusive; this interactive debug request does not use `-x`.

The Super model, train data, and validation data are supplied explicitly. The
launcher runs the current checkout so uncommitted local recipe changes are not
lost during preparation.

Dry-run the exact LSF submissions:

```bash
MODEL_PATH=/proj/.../NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
TRAIN_PATH=/proj/.../rlvr1/train-split.jsonl \
VAL_PATH=/proj/.../rlvr1/val-split.jsonl \
CONTAINER_IMAGE=/proj/.../nemo-rl-v0.7.0.sif \
bash examples/bluevela/grpo_super_120b_27n8g/submit.sh --dry-run
```

Submit only after reviewing the dry-run:

```bash
MODEL_PATH=/proj/.../NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
TRAIN_PATH=/proj/.../rlvr1/train-split.jsonl \
VAL_PATH=/proj/.../rlvr1/val-split.jsonl \
CONTAINER_IMAGE=/proj/.../nemo-rl-v0.7.0.sif \
bash examples/bluevela/grpo_super_120b_27n8g/submit.sh --submit
```

The launcher does not download or rewrite the dataset. Preparation validates
the supplied model/data paths and prefetches NeMo-Gym environments. Outputs,
cache, and checkpoints default below:

```text
/proj/datasets/interns/yuetai/agent_envs/more_task/
```

Monitor with `bjobs -l PREP_JOB_ID TRAIN_JOB_ID`; the run directory is printed
by `submit.sh` and contains the driver, Ray-node, and preparation logs.
