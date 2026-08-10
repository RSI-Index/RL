# Blue Vela 集群使用说明（`more_task/envs`）

这份说明供共用 `yuetai` 账号的开发者使用。双方看到相同的 HOME、GPFS、LSF 作业、缓存和凭据状态，因此不需要复制 SSH key 或重新配置文件权限；但必须隔离各自的作业和输出，避免覆盖或误杀对方正在运行的任务。

## 1. 目录与存储约定

- 源码和小型配置放在：`/u/yuetai/more_task/envs`
- 可参考的已验证环境在：`/u/yuetai/more_task/post_training/RL/examples/bluevela/grpo_nanov3_30ba3b_32n8g`
- `more_task` 的大型数据、镜像、checkpoint 和运行输出建议统一放在：
  `/proj/datasets/interns/yuetai/agent_envs/more_task`

HOME 空间只用于源码和小文件。模型、数据集、Hugging Face cache、SIF、日志和 checkpoint 必须放在 `/proj/datasets/interns/yuetai/...`。`/proj` 没有回收站，不要对共享根目录执行递归删除。

建议在作业脚本里显式设置：

```bash
export HF_HOME=/proj/datasets/interns/yuetai/hf-home
export HF_DATASETS_CACHE=/proj/datasets/interns/yuetai/hf-datasets-cache
export RUN_ROOT=/proj/datasets/interns/yuetai/agent_envs/more_task
mkdir -p "$RUN_ROOT"
```

检查配额：

```bash
gquota ~
gquota /proj/datasets
```

## 2. 登录节点的边界

登录入口是 `login{1,2,3,4}.bluevela.rmf.ibm.com`，常用入口：

```bash
ssh yuetai@login3.bluevela.rmf.ibm.com
```

登录节点只用于编辑代码、查看日志、Git 操作和轻量检查。以下工作必须提交到 LSF compute node：

- GPU 训练或推理；
- 大规模数据预处理；
- 编译 CUDA 扩展；
- 构建 Apptainer/SIF；
- 长时间或高内存 CPU 任务。

不要直接 SSH compute node；需要交互环境时使用 LSF interactive job。Blue Vela 没有常规 module/lmod 工作流，软件主要位于 `/opt/share` 或用户自己的环境中。

## 3. LSF 基本规则

Blue Vela 使用 IBM Spectrum LSF，不是 Slurm。常用命令：

```bash
bjobs
bjobs -a
bjobs -l <JOB_ID>
bqueues -u
blimits -w -a
monitor-bv -u grp_models
bkill <JOB_ID>
```

只有确认 job ID、job name 和输出目录确实属于自己当前任务后才能执行 `bkill`。共用账号时，不要按模糊名称批量杀作业。

每次提交必须显式指定：

- queue：通常为 `normal`；
- fair-share group：`grp_models`；
- memory：不写 `-M` 时默认内存很小，容易 OOM；
- walltime（需要时）；
- GPU 数和 exclusive mode；
- 独立的 stdout/stderr 路径。

### 单节点 GPU 作业模板

先创建唯一的 run 目录，再提交：

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-<你的名字或任务标签>"
RUN_DIR="/proj/datasets/interns/yuetai/agent_envs/more_task/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
```

作业脚本示例：

```bash
#!/usr/bin/env bash
#BSUB -J mt-<task>-<owner>
#BSUB -G grp_models
#BSUB -q normal
#BSUB -n 32
#BSUB -gpu "num=8/task:mode=exclusive_process"
#BSUB -M 256G
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -o /absolute/run/path/lsf.%J.out
#BSUB -e /absolute/run/path/lsf.%J.err

set -euo pipefail

hostname
nvidia-smi

# 在这里执行训练或评估命令。
# 命令完成后必须检查预期产物，避免 payload 失败但 driver 仍返回 DONE。
test -s /absolute/run/path/result.json
```

提交：

```bash
bsub < job.sh
```

注意：`#BSUB` 行不会展开 shell 变量。日志路径要写成提交时已经确定的绝对路径；如果需要动态参数，使用 `bsub` CLI 参数或在脚本正文中处理。

### 交互式 GPU 作业

```bash
bsub -q interactive -Is -n 8 -M 64G -G grp_models \
  -gpu "num=1:mode=exclusive_process" bash
```

交互调试结束后及时退出。一般长任务应改成普通 batch job，并把日志与结果写到
GPFS；第 10 节的大规模 RL 作业例外，必须按目标 config 一次性申请完整
interactive allocation，并在该 allocation 内持续调试和运行。

### 多节点作业

超过一个 8-GPU 节点时，使用 `LSB_MCPU_HOSTS` 获取分配到的 host，并通过 LSF 的 `blaunch` 启动远端进程。现有的 4×8 H100 示例可参考：

- `/u/yuetai/more_task/envs/samples/gated_deltanet/environment/launch_gdn_4x8_lsf.sh`
- `/u/yuetai/Scale_AutoResearch/envs/tasks/gated_deltanet/environment/launch_gdn_4x8_lsf.sh`

不要直接复制示例中的节点数、训练预算或数据路径；只参考 host discovery、`blaunch` 和 `torchrun` 的调度方式。多节点脚本必须确保远端进程退出后，主作业能正确收集 return code 并清理残留 worker。

## 4. Apptainer/SIF

当前项目验证过的 Apptainer 路径是：

```bash
export APPTAINER_ROOT=/proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env
export PATH="$APPTAINER_ROOT/bin:$PATH"
export APPTAINER_BIND=/etc/resolv.conf
apptainer --version
```

运行 GPU 容器：

```bash
apptainer exec \
  --nv \
  --containall \
  --writable-tmpfs \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
  --bind /proj:/proj \
  /absolute/path/to/image.sif \
  python /absolute/path/to/program.py
```

关键约束：

- GPU 容器必须使用 `--nv`；
- `--containall` 会丢弃部分宿主环境变量，GPU pinning 等变量要用 `--env` 或 `APPTAINERENV_*` 显式传入；
- `APPTAINER_BIND=/etc/resolv.conf` 用于避免 `--containall` 下 DNS 异常；
- 共享文件必须显式 bind，且输入资产尽量使用 `:ro`；
- 容器内 `/opt`、`/usr` 的临时 overlay 改动不会自动成为其他节点可见的共享状态。

### 构建 SIF

不要在 login node 构建，也不要直接在 GPFS 上完成 squashfs 打包。应提交 CPU job，在 compute node 的本地 `/tmp` 中构建，成功后再复制到 GPFS。可直接参考：

- `/u/yuetai/Scale_AutoResearch/envs/lsf_validation/build_one.sh`
- `/u/yuetai/Scale_AutoResearch/envs/lsf_validation/defs/`

构建 job 应请求足够的内存和本地临时空间。build-time smoke 只能做不依赖 GPU 的包/metadata 检查；真实 CUDA import 和模型加载应放到第一个 GPU smoke job。

## 5. Harbor 环境

Harbor task 的样例位于：

- `/u/yuetai/more_task/envs/samples/gated_deltanet`
- `/u/yuetai/more_task/envs/samples/rlm`

Blue Vela 上不能假设 Docker daemon 可用。GPU Harbor run 使用预构建 SIF 和自定义 LSF/Apptainer backend。实现和完整运行方式参考：

- `/u/yuetai/Scale_AutoResearch/envs/envs_backend/lsf_apptainer.py`
- `/u/yuetai/Scale_AutoResearch/envs/lsf_validation/README.md`
- `/u/yuetai/Scale_AutoResearch/envs/lsf_validation/submit_full_run.sh`

这些 Scale AutoResearch 脚本包含 task-specific 的资源数、数据资产、policy 和审计规则。新任务可以复用集群适配模式，但不能把现有 task 的科学预算或 verifier 契约原样套用。

在真正占用 GPU 前，优先提供并运行 read-only preflight 或 dry-run，打印最终 `bsub` 命令、镜像路径、bind 路径、GPU 数、内存和输出目录。

## 6. 共用账号的协作规则

1. Job name 使用 `mt-<task>-<owner>`，不要使用含义不清的 `test`、`run`。
2. 每次运行使用新的 UTC timestamp run directory；不要复用或覆盖他人的 `logs/`、`output/`、`checkpoint/`。
3. 提交前执行 `bjobs`，确认没有同名任务或资源冲突。
4. 修改共享 SIF、数据 manifest、公共 cache 或 launcher 前，先确认没有 live job 正在读取它。
5. 不要修改 live run 的冻结 `control/`；修复后创建新 run，或者使用该 launcher 明确支持的 resume 流程。
6. 不要删除不属于自己当前任务的 job、日志、checkpoint、cache 或临时目录。
7. 大作业提交后记录 job ID、run directory、代码 commit/diff、镜像 SHA256 和预期产物。

建议在每个 run 目录保存一个简短的 `RUN_INFO.md` 或 JSON manifest，至少记录：

```text
owner/tag:
purpose:
submitted_at_utc:
lsf_job_id:
source_commit:
image:
gpu_count:
expected_outputs:
```

## 7. 凭据与网络

虽然双方共用账号和现有登录状态，也不要把以下内容复制到 task、镜像、日志或 run artifact：

- `~/.ssh`；
- Codex `auth.json`；
- API keys、tokens、cookies；
- 个人访问凭据或内部 endpoint secrets。

只把凭据提供给确实需要它的控制面或 verifier。候选/agent 容器默认不应获得宿主凭据。Apptainer 共享宿主网络 namespace，因此 `network_mode=no-network` 不能被当前 backend 物理保证；需要离线实验时，应同时使用本地资产、不给容器凭据、关闭相应网络工具并做运行后审计。

## 8. 常见故障

- **作业立刻 OOM/exit 135**：通常是忘了设置 `-M`。
- **作业显示 DONE，但没有结果**：driver 没有启用 `set -euo pipefail`，或没有检查最终产物。
- **容器看不到 GPU**：检查 LSF 是否分配 GPU，以及 Apptainer 是否带 `--nv`。
- **多个进程都跑到 GPU 0**：`--containall` 丢失了 `CUDA_VISIBLE_DEVICES`；使用 `--env` 或 `APPTAINERENV_CUDA_VISIBLE_DEVICES`。
- **SIF 在 GPFS 上构建失败或空间不足**：在 compute node 的本地 `/tmp` 构建，再复制成最终 SIF。
- **build 阶段报 CUDA driver/triton 错误**：不要在无 GPU build node 上做真实 GPU import。
- **多节点远端 worker 没有退出**：检查 `blaunch`/`torchrun` 的信号传播、return code 和清理逻辑。
- **GPFS 上 `np.memmap(mode='w+')` 出现 SIGBUS**：改成普通流式写入。

## 9. 开始工作前的最小检查

```bash
cd /u/yuetai/more_task/envs

which bsub
which bjobs
bqueues -u
bjobs

test -x /proj/datasets/interns/yuetai/rsi-nemotron/apptainer-env/bin/apptainer
gquota ~
gquota /proj/datasets
```

确认上述检查通过，一般任务再先提交一个短 CPU/GPU smoke。第 10 节的大规模
RL 作业不得为 smoke 单独申请较少节点；应直接申请 config 要求的完整
interactive allocation，并在其中完成 smoke、debug 和正式运行。

## 10. 大规模 RL 多节点作业基线（参考 Nano v3）

大规模 RL 不应直接套用普通单节点 GPU 模板。提交配置、资源申请、Ray
拓扑和通信链路必须一起检查。推荐以
`examples/bluevela/grpo_nanov3_30ba3b_32n8g/` 为参考实现：

- `submit.sh`：创建唯一 run 目录并提交 preparation/training LSF job；在本节
  的 interactive 调试流程中只用 `--dry-run` 核对资源，不执行 `--submit`；
- `prepare.sh`：在 compute node 上准备 SIF、模型、数据、NeMo-Gym 和
  commit-scoped vLLM 环境；
- `train.sh`：校验 LSF 分配，通过 `blaunch` 启动 Ray head/worker，并在
  容器内启动训练；
- `ray_node.sh`：配置 Ray 节点、NCCL 和 RDMA 设备；
- `../nemo_gym/grpo_nanov3.yaml`：训练配置中的 `cluster.gpus_per_node: 8`
  和 `cluster.num_nodes: 32`。

### 10.1 必须开启 W&B

完整规模的多节点 RL 作业必须开启 W&B；只写终端日志、TensorBoard 或本地
文件不算满足基线。W&B 是启动验收项，不是训练出问题后才补开的可选项。
最终生效配置至少应包含：

```text
logger.wandb_enabled=true
logger.monitor_gpus=true
logger.wandb.project=nemo-rl
logger.wandb.name=$RUN_ID
```

其中 `RUN_ID` 必须全局唯一，并同时用于 W&B run name 和本地 run 目录，确保
每次 attempt 的曲线、日志和 checkpoint 能一一对应，并能追溯到同一个
interactive allocation job ID。Nano v3 launcher 会强制覆盖
`logger.wandb_enabled=true`，从 mode-600 的 `WANDB_API_KEY_FILE` 读取凭据，
并把 W&B URL 写入训练成功标记；不得把 API key 写进脚本、配置、命令行或
run 目录。

提交前确认 secret 文件存在且权限正确；训练启动后，应在第一个训练 step
之前确认 `train-driver.log` 已出现 W&B run URL，且页面持续收到训练指标和
每节点 GPU 利用率。至少关注 reward、loss、KL/entropy、生成长度、各阶段
耗时/吞吐和 GPU 利用率/显存。若 W&B 初始化或鉴权失败、run name 重复、
页面没有 step 前进，或 GPU 监控缺失，应停止作业并修复，不能让完整规模
训练在不可观测状态下继续。完整 NeMo-Gym result table 可能很大，可以像
Nano v3 recipe 一样设置
`logger.wandb.log_nemo_gym_full_result_tables=false`；这不等于关闭标量指标和
GPU 监控。

### 10.2 只申请一次 interactive allocation，并持续在原节点调试

本节的大规模 RL 调试采用比前文通用 batch 建议更严格的规则：开始 debug
或实际运行前，必须先解析目标 config 及全部命令行 override 的最终有效值，
令 `N = cluster.num_nodes`、`G = cluster.gpus_per_node`，再通过 LSF 一次性
申请 `N` 个 interactive nodes、每节点 `G` 张 GPU，以及匹配的 CPU 和内存。
不得默认写死为 32 个节点，也不得先申请较少节点做 smoke 再扩容；Nano v3
只是 `N=32`、`G=8` 的参考实例。进入 allocation 后，记录 LSF job ID 和
host list；后续 smoke、排障、修改配置、重启 Ray、重跑训练命令和正式运行
都留在这批原始节点内完成。

只要该 interactive allocation 仍然存活，就不得再次调用 `bsub`，也不得
再次执行会提交新 LSF job 的 `submit.sh --submit`。`submit.sh --dry-run`
可以用来核对最终资源 shape，但不能把反复提交 batch job 当作调试循环。
某次进程失败后，应保留当前 allocation 和故障现场，在原节点内停止残留的
Ray/NCCL/训练进程，修复后直接重新运行；不要为了获得“干净环境”换一批节点，
否则会丢失节点、网络和拓扑相关问题的现场。

因此，用于 interactive 调试的 launcher 必须提供“使用当前 LSF allocation
直接运行”的入口。若现有脚本只有 batch submit 模式，应先补这个入口，不能
通过连续重交 job 绕过。每次训练 attempt 仍应使用可区分的 attempt/run ID，
将日志和 W&B 曲线分开，但它们必须记录同一个 allocation job ID 和 host
list。原 allocation 已退出、到达 walltime 或被 LSF 回收时，不得自动重交；
先记录退出原因和未完成状态，再由操作者决定后续动作。

### 10.3 先保证每节点 CPU 和内存

节点数和 GPU 数必须以最终生效的 config 为准：LSF host 数为 `N`，每节点
GPU 数为 `G`，GPU 总数为 `N * G`。下面只是 Nano v3 的 `N=32`、`G=8`
资源算例，不能把 32 和 256 照搬到其他 config：

| allocation | 节点数 | 每节点 CPU slot | 每节点内存 | 其他关键资源 |
| --- | ---: | ---: | ---: | --- |
| Nano v3 interactive | 32（来自 config） | 64 | 524288 MB（约 512 GiB） | 总计 2048 slot、每节点 8 张 GPU |

对应的 Nano v3 interactive LSF 约束包含：

```text
-Is
-n 2048
-R "span[ptile=64] rusage[mem=524288]"
-gpu "num=8:mode=shared:j_exclusive=yes"
-x
```

这里 `span[ptile=64]` 保证每个节点得到 64 个 CPU slot；`-x` 让整个
training 节点独占，避免 CPU-only job 与训练共享节点；
`j_exclusive=yes` 保证申请到的 GPU 不被其他 job 共用。脚本还会检查
`LSB_MCPU_HOSTS` 中的节点数和每节点 slot 数、每个节点是否有 8 张 H100，
不满足时在训练开始前失败。这些数字来自 Nano v3 config；其他 config 必须
用自己的 `N` 和 `G` 做同样校验。

提交前必须先执行 dry-run，确认最终渲染出的 `-n`、`-M`、`-R`、GPU 数和
节点数，而不是只看队列里显示的 job name：

```bash
cd /u/yuetai/more_task/post_training/RL
bash examples/bluevela/grpo_nanov3_30ba3b_32n8g/submit.sh --dry-run
```

特别注意：多节点 RL 的 CPU slot 和内存是“每节点”资源。`TRAIN_HOSTS`
必须等于最终 config 的 `N`；以 Nano v3 的每节点 64 slot 为例，
`TRAIN_SLOTS` 必须为 `N * 64`，预期 GPU 总数必须为 `N * G`。每节点 CPU、
内存和 GPU 需求也应从目标 recipe 推导并在启动前校验，不能只改节点数。
preparation 不再单独提交 1-node job，而是在这次完整 interactive allocation
内运行；allocation 的 walltime 必须覆盖 preparation、smoke、debug 和正式
训练，head node 还要有足够的本地 `/tmp`。

### 10.4 多节点通信必须使用 RDMA

节点内 GPU collective 应使用 NVLink，跨节点 collective 必须走
InfiniBand verbs/RDMA；不能用普通 TCP socket 作为 NCCL 的数据通道。以
Nano v3 recipe 为准，下面几项必须同时存在：

1. 宿主机有 `/dev/infiniband/*`，容器使用 `--nv` 并显式 bind
   `/dev/infiniband:/dev/infiniband`；
2. 容器内能看到 `/sys/class/infiniband/*`；
3. `NCCL_IB_DISABLE=0`，并设置与 Blue Vela 拓扑匹配的
   `NCCL_IB_HCA`；
4. `NCCL_DEBUG=INFO` 和 `NCCL_DEBUG_SUBSYS=INIT,NET` 保持开启，便于核对
   实际 transport；
5. `ray_node.sh` 使用 `blaunch` 在 LSF 分配的所有节点启动 Ray，且所有
   worker 连接同一个 head address。

该参考 recipe 当前使用：

```bash
NCCL_IB_HCA='^=mlx5_1,mlx5_6'
NCCL_IB_DISABLE=0
```

`NCCL_SOCKET_IFNAME` 在该脚本中指定 Blue Vela 的 IPoIB interface，用于
NCCL/Ray 的 TCP bootstrap 或 socket fallback 选择；它不是 RDMA 开关，
也不能用来替代 `NCCL_IB_HCA`。Ray 控制面使用 TCP address（当前 head
address 为 `ibp26s0` 上的 IP 加端口 `1200`）是正常的；需要禁止的是
NCCL collective 回退到 `NET/Socket`。训练日志中应看到
`NCCL ... NET/IB`（或等价的 IB transport 信息），若出现 collective 使用
`NET/Socket`、`NCCL_IB_DISABLE=1`、“RDMA devices are missing”等信息，
应立即停止该训练进程，在原 interactive allocation 和同一批节点内检查
HCA、容器 bind 和 NCCL 环境，修复后直接重跑，不得重新提交 LSF job。

可在分配到节点后用下面的检查确认基础条件；不要在 login node 上把这些
检查结果当成 training 节点已经就绪的证明：

```bash
nvidia-smi
nvidia-smi topo -m
ls -l /dev/infiniband
ip -4 addr show ibp26s0
```

`nvidia-smi topo -m` 应能看到同节点 GPU 之间的 NVLink 拓扑；跨节点的
NCCL 日志则应显示 IB/RDMA transport。两者分别对应节点内和节点间链路，
不能因为 Ray 控制面能连通就认为 GPU collective 链路正确。

训练启动后重点检查：

```bash
rg -n 'NCCL.*NET/(IB|Socket)|NET/(IB|Socket)|RDMA|Infiniband' \
  RUN_DIR/logs/train-driver.log RUN_DIR/logs/ray/
```

若新增或修改 launcher，不得删除 `--nv`、`/dev/infiniband` bind、
`NCCL_IB_DISABLE=0` 或 HCA allowlist，也不要用只设置
`NCCL_SOCKET_IFNAME` 的 socket-only 配置替代它们。修改通信配置后，必须
在已经申请到的完整 `N` 节点 interactive allocation 内做短 smoke，确认
全部节点的 NCCL 日志使用 IB，再在同一 allocation 内继续正式训练。

### 10.5 固定 run 身份和所有输入

同一个 interactive allocation 内的每次训练 attempt 都必须创建新的、不可
复用的 `RUN_DIR`，并在启动时固定以下信息：

- 完整的 40 位 source commit；
- 容器路径及内容 hash；
- 模型、tokenizer 和数据版本；
- 实际训练 entrypoint、config 及所有命令行 override；
- interactive allocation job ID、host list、节点/GPU shape、checkpoint 路径和
  W&B run name。

Nano v3 launcher 会拒绝复用已有 `RUN_DIR`，复制本次实际执行的 control
scripts，生成 `control/payloads.sha256`，并把上述主要字段写进
`RUN_INFO.txt`。修改 launcher 时应保留这些行为。不要直接从可变 branch
或登录节点当前 worktree 启动目标规模作业，也不要在启动后原地修改 run
目录里的脚本或配置；需要修改时创建新的 attempt/run，并让 W&B 名称同步
变化，但继续使用同一个 interactive allocation。

### 10.6 checkpoint 必须覆盖队列时限和续训

完整规模作业必须启用 checkpoint，保存恢复训练所需的 model、optimizer、
dataloader、epoch 和 step 状态。保存目录应位于持久存储，不能放在 compute
node 的本地 `/tmp`。保存周期既要能限制故障重算量，也不能频繁到明显阻塞
训练。

checkpoint deadline 必须早于 LSF walltime，并预留足够的集体写盘和清理
时间。Nano v3 的 4 小时 allocation 使用：

```text
checkpointing.enabled=true
checkpointing.save_period=10
checkpointing.checkpoint_must_save_by=00:03:40:00
checkpointing.save_optimizer=true
```

同一个 interactive allocation 内续训时，应使用新的 attempt ID 和 W&B run，
同时显式指向上一次的 `CHECKPOINT_DIR`，但不得重新提交 LSF job。在完整
`N` 节点 allocation 内开始长时间训练前，至少做一次 save/resume smoke，
确认能发现最新 `step_*` checkpoint，恢复后的 step/epoch 正确，且 W&B
曲线能通过 run 名称和 checkpoint 路径追溯到上一段训练。不能只看到
checkpoint 目录存在就认为它可恢复。

### 10.7 按 config 一次申请完整 allocation，并在其中验收

不要先申请少量节点做 smoke，再释放后申请完整规模。正确顺序是：

1. 解析最终 config 和 overrides，得到 `N = cluster.num_nodes`、
   `G = cluster.gpus_per_node`；
2. `submit.sh --dry-run`，核对路径、commit、每节点资源、`N` 个节点和
   `N * G` 张 GPU；
3. 直接申请一次 `N` 节点、每节点 `G` 张 GPU 的 interactive allocation；
4. 进入该 allocation 后完成 preparation，并校验模型、数据、容器和 runtime；
5. 在全部 `N` 个节点上启动 Ray，做短 smoke，验证 W&B、NCCL/RDMA、一次
   训练 step 和一次 checkpoint；
6. smoke 通过后不退出 allocation、不更换节点、不重新提交 LSF job，直接在
   同一批节点上继续正式训练和后续 debug。

完整作业开始训练前，应同时确认 Ray 注册了全部 `N` 个节点和 `N * G` 张
GPU、恰好有 `N` 个 node ready markers、NCCL 使用 IB、W&B run 已在线且
没有 NaN/Inf。`bjobs` 显示 `RUN` 或 Ray head 能连接，都不能单独证明训练
健康。任一节点/actor 缺失、collective 回退到 socket、指标长期没有 step
前进或首个 checkpoint 写入失败时，应立即终止训练进程但保留 interactive
allocation，先保存日志和故障节点列表，再在同一批节点上排障和重跑；不要
重新提交 LSF job，也不要用降节点数、跳过校验或关闭监控的方式继续。

### 10.8 以状态标记和可恢复产物判定成功

LSF job 退出并不等于训练成功。以 Nano v3 recipe 为例：

- preparation 只有生成 `status/PREP_SUCCESS` 才算完成；
- training 只有 driver 返回 0 且生成 `status/SUCCESS` 才算完成；
- `status/PREP_FAILED`、`status/TRAIN_FAILED` 或任一 Ray node failed marker
  都必须按失败处理；
- 成功标记中应能核对 job ID、source commit、训练 shape 和 W&B URL；
- 结束后保留 `RUN_INFO.txt`、control payload/hash、driver/Ray/NCCL 日志、
  W&B run 和最后一个可恢复 checkpoint。

即使 W&B 曲线看起来正常，只要缺少成功标记或可恢复 checkpoint，也不要
删除 run 目录或宣布完成。反之，成功标记存在但 W&B URL 缺失、节点规模不
符或关键指标不完整，也应先审计日志和产物。

项目               之前版本       当前版本
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━
   节点                     32             32
  ──────────────  ─────────────  ─────────────
   GPU/节点        8，GPU 独占    8，GPU 独占
  ──────────────  ─────────────  ─────────────
   CPU/节点           17 slots       64 slots
  ──────────────  ─────────────  ─────────────
   内存/节点           512 GiB        512 GiB
  ──────────────  ─────────────  ─────────────
   整机独占            没有 -x          有 -x
  ──────────────  ─────────────  ─────────────
   总 CPU slots            544           2048
