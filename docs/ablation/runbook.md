# 可复现运行入口

在原仓库规定的 Isaac Lab `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` 原生环境中，从仓库根目录运行。先按原 README 准备 G1 MJCF/USD 和完整40段数据。实验代码不要求改动原训练入口。GPU、软件构建、资产和 motion SHA 会保存在每次运行的 manifest。

## 算法测试

```bash
python -m unittest discover -s tests -v
```

测试中的小张量与线性网络是算法回归测试，不产生物理实验结果。

## 原生资格与最小实验

```bash
python tools/test_env.py --num_envs 32
python tools/test_reset.py
python -m experiments.ablation.run_suite \
  --phase smoke --output logs/ablation/smoke-v1 \
  --gpu 1 --max-gpu-hours 0.75 --seeds 0
```

Smoke沿用完整网络、原生仿真器和完整18段评估，只将训练批量、迭代数和imagination horizon降到小规模（H=4）；不据此评价效果。各条件3次迭代，预训练1次；各方法初始/最终checkpoint物理评估。它必须验证代码分支都执行、参数确实更新、冻结WM不变、所有指标有限，不要求能跟踪动作。

共7次训练进程：1次独立pretrain、PPO-Sim、在线PPO-WM、SuperMimic、硬终止冻结PPO-WM，以及软终止的冻结PPO/BPTT配对。每个seed完整执行6种policy条件，不把不同终止目标的配对混为同一机制实验。

输出目录禁止覆盖；失败日志保留。检查 `campaign.json`/每个 `completion.json`，有失败或达到预算不自动无限重试。

## H=32 短程曲线

先据Smoke的显存、编译和吞吐测量调整有限预算，再启动：

```bash
python -m experiments.ablation.run_suite \
  --phase pilot --output logs/ablation/pilot-v1 \
  --gpu 1 --max-gpu-hours 2 --seeds 0
python -m experiments.ablation.analyze \
  --evaluations logs/ablation/pilot-v1/*/ckpt_*/evaluation.json \
  --campaign logs/ablation/pilot-v1/campaign.json \
  --output results/ablation/pilot-v1
```

`pilot` 是256环境/256imagined batch，100迭代、WM预热30迭代，k=8/H=32，原始宽度网络与固定FP32/TF32；不是论文训练预算。PPO-WM frozen 与BPTT frozen都从同一个 `pretrain-seed0/bootstrap.pt` 开始；在线模型与预训练账本分开。比较PPO-WM和PPO-Sim请主要看physical return对总成本，不能用迭代数判快慢。

## 正式矩阵

`experiments/ablation/configs/matrix.json` 给出种子0/1/2及18/40段范围。formal JSON是待资格确认的候选配置，不是已运行结果。必须先在目标4096环境规模做一至数次完整更新的显存/时间测量，保存forecast和来源。runner对formal要求 `--forecast`，且预计总GPU-hours不超过显式预算、预计峰值显存低于可用显存的85%。

```bash
python -m experiments.ablation.run_suite \
  --phase formal --output logs/ablation/formal-locomotion-v1 \
  --gpu 1 --max-gpu-hours BUDGET --seeds 0 1 2 \
  --forecast measured-forecast.json
# 40段单独运行/分析，不能与18段曲线混合
python -m experiments.ablation.run_suite \
  --phase formal --subset all40 --output logs/ablation/formal-all40-v1 \
  --gpu 1 --max-gpu-hours BUDGET --seeds 0 1 2 \
  --forecast measured-forecast-all40.json
```

`BUDGET` 必须替换为基于实测并获授权的数值。forecast包含 `gpu`、`subset`、`projected_campaign_gpu_hours`、`projected_peak_memory_mib`，并应保留原始计时日志和配置身份；手填一个通过门槛的数字不能代替资源资格。

## 单方法与独立复评

```bash
python -m experiments.ablation.train \
  --config experiments/ablation/configs/pilot-ppo_sim.json \
  --seed 0 --output logs/ablation/custom-ppo-sim
python -m experiments.ablation.evaluate \
  --checkpoint logs/ablation/custom-ppo-sim/ckpt_000100.pt \
  --output logs/ablation/custom-ppo-sim/evaluation-repeat1
```

重复评估另存，不混入主曲线充当额外seed；分析器会拒绝同method/seed/iteration的重复记录。

## 固定库的Prediction / Gradient Probe

先选择一个固定collector checkpoint及至少两个含WM的训练阶段checkpoint。每次probe默认64个独立起点、4个随机方向+1个预测梯度方向、4个尺度；每个FD方向要做多次真实物理rollout，先用 `--samples 4 --directions 1` 测成本。

```bash
python -m experiments.ablation.fidelity \
  --collector logs/ablation/pilot-v1/supermimic-seed0/ckpt_000100.pt \
  --checkpoints logs/ablation/pilot-v1/supermimic-seed0/ckpt_000025.pt \
                logs/ablation/pilot-v1/supermimic-seed0/ckpt_000100.pt \
  --samples 4 --directions 1 --epsilons 0.003 0.01 0.03 0.1 \
  --output logs/ablation/fidelity-smoke-v1
```

后续checkpoint用 `--bank .../heldout-bank.pt`，`--samples`与原库一致。库从不参与模型训练。`prediction.csv`与`gradient.csv`保存逐样本、checkpoint、horizon/方向/尺度/接触标签与数值参考有效性。`physical-baseline.pt`保留完整物理状态序列和contact signatures。具体数学目标、单位、噪声筛选与证据边界见 [protocol.md](protocol.md)。

对同一库的稳定接触短窗口，另外运行 `--gradient-horizon 8`（必要时1），输出到新目录；不同梯度horizon单独分析。缩短梯度窗口不会缩短1/8/16/32步状态预测评估。

```bash
python -m experiments.ablation.plot_fidelity logs/ablation/fidelity-smoke-v1 \
  --output results/ablation/fidelity-smoke-v1 \
  --evaluations logs/ablation/pilot-v1/supermimic-seed0/ckpt_*/evaluation.json
```

每轮campaign运行期间保持源码和配置不变；runner会在每个进程前核验快照并拒绝继续混用代码版本。修改后必须新建campaign目录。安装/资产资格、被排除的运行和独立gradient probe分别保留成本，不塞进某个方法的训练收益。
