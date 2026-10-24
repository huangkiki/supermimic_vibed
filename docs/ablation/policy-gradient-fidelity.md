# 完整 Policy Parameter Gradient：实现与证据边界

本阶段验证的是：固定 Policy、WM、Critic、起点和随机数后，原 SuperMimic 策略目标对**策略参数**的梯度，在物理仿真中是否给出可靠方向，以及实际 Adam 更新是否有效。

**已完成 15 组 MuJoCo-Warp 参数梯度探针**：H1/8/32 的四种 Critic/直接动作梯度组合，以及 3 组 FP32 检查；累计 0.169490 GPU-hours。只覆盖一个 checkpoint、两个 run 起点。[结果、原始数据与局限](policy-gradient-native-v1.md)。正式队列在 GPU0 继续，本批使用独立 GPU2 与 checkout，未修改正式执行源码。

## 源码审计与目标

`supermimic/train.py::policy_update()` 的 actor 部分包含：原始 replay history 构造、观测噪声、闭环 Policy、逐步 WM、dense reward、soft termination、折扣和末端 Critic Bootstrap。策略损失为

\[
L(\theta)=-\operatorname{mean}_b\left[\sum_{t=0}^{H-1}\gamma^t c_t r_t+\gamma^H c_H V(s_H)\right],\quad
c_t=\prod_{j<t}(1-d_j).
\]

Critic 参数冻结，但其输入的状态梯度、存活权重梯度仍回传至 Policy。原函数随后执行 gradient norm clipping（1.0）、value clipping（0.1）和 Adam；**策略更新之后**才更新 Critic 归一化统计并训练 Critic。

有三个影响解释的已有差异：

1. 原 `fidelity.py` 固定动作序列、不含 Critic，只回答 Action Gradient 问题。
2. `experiments/ablation/train.py::imagine()` 在 rollout 前更新 Critic 的初始输入归一化统计。原训练入口的时间顺序不同；本阶段不改任何训练算法，也不重写正在运行的实验。
3. 通常的 checkpoint 有 Policy/WM/Critic/Adam，却没有当时 minibatch 和 RNG；旧 held-out bank 只存处理后的观测历史，没有原始 replay history。

因此，新诊断的准确表述是：**在已有 checkpoint 上，运行原训练函数的 actor 计算路径，计算一次固定起点的诊断更新**。它不是对过去某次训练更新的精确回放。

## 实现

- [`policy_fidelity.py`](../../experiments/ablation/policy_fidelity.py) 解析本仓库可信 `train.py` 的 AST，仅提取 helper 和 `policy_update` 的 actor 前缀；不执行模块级仿真启动，不复制 rollout/目标/clipping 公式。保留源文件和提取代码 SHA256。执行 eager 路径，是否与编译后的 CUDA 数值一致仍待实测。
- `capture()` 深拷贝 Policy 和历史 Adam，在原语句执行期间保存原始 loss gradient、norm-clipped gradient、最终 value-clipped gradient、对应 ascent gradient、实际 Adam 参数位移和更新前后 optimizer state。缺失历史 moments 会拒绝运行。actual Adam 位移与归一化的 Adam 方向分开记录。
- `ClosedLoopPhysical` 仅把 `_imag_step` 使用的 dynamics 换成原生 `TrackingSim.step()`，继续使用原闭环 Policy、观测噪声、history 更新、reward、soft termination 和 Critic。每次恢复相同物理状态、仿真参数和 solver warm start。
- 物理 rollout 始终走完 H 步以构造匹配的软终止代理目标；另用原生 `done` 的前序存活 mask 累计硬终止物理 Return，包含第一次失败步的奖励。两者均为固定 H 的折扣和，**不等于全片段 benchmark 的归一化 tracking score**。
- 原 bank 只读；必要时重放已保存 prefix，恢复 raw state/action/xi/frame history。只有末态误差 ≤1e-5 且 frame/参数/上一动作一致才接受，另存 `policy-bank.pt`。明确标记这是数值重建，不是历史训练 minibatch；重建失败不以复制当前观测补齐历史来绕过。
- 所有扰动共享固定观测噪声；调用结束恢复外部 CPU/所选 GPU RNG、参数和 requires-grad flags。归一化统计不更新。

## 两种消融的解释

`--no-critic` 删除末端 Bootstrap，物理代理目标也删除相同项。它改变目标，结果应单独报告。

`--detach-action-reward` 只对 `mdp.reward(S,f,a,a_prev)` 中显式动作参数 `a`、`a_prev` 停止梯度；状态仍保留 WM→动作→Policy 梯度，soft termination 与 Critic 路径保留。**前向 reward 不变**，和旧 `state_only` 删除 action-rate reward 的实验不同。该消融是 surrogate update，不能再把其梯度与完整 forward objective 的差分称为同一目标的准确率；代码将其标记为方向有效性诊断。

## 方向、有限差分与统计

对 raw ascent、clipped ascent、historical Adam 方向及固定随机方向，分别单位 L2 归一化后扫描 ε；同时保存未归一化原始向量和实际 Adam 步的物理改善。记录浮点加法后实际参数位移，不能仅依赖名义 ε。

方向导数预测为固定 minibatch **平均目标**的 `gradient · direction`；物理中央差分也是相同 minibatch 的平均目标。逐起点 return 保留在张量文件中，但不把一个批次梯度复制成每个起点的独立预测。每个 ε、目标和方向家族分别报告 sign agreement/correlation，不混合尺度，不把两次复评当两个训练 seed。

至少两次相同基线运行估计 numerical noise；默认三次。沿用数值解析门槛：信号超过重复噪声与绝对下限、相邻 ε 的导数相对差异 ≤25%。这只是敏感性筛查，中央差分不是可信解析 Jacobian，也不是绝对真值。硬终止目标与 WM 目标不同，其符号一致性仅用于物理方向有效性分析。

## 最小运行与资源预算

CPU 回归检查（已执行）：

```bash
python -m unittest discover -s tests -p test_policy_fidelity.py -v
python -m unittest discover -s tests -v
python -m experiments.ablation.fidelity --probe-space policy --help
```

原路径新增 15 项测试；新增分析与原生数据复算 8 项测试。覆盖 H1/8/32 与完整原函数的 loss/Adam 位移逐值一致、光滑闭环中央差分、两次裁剪、非零历史 moments、同前向奖励的 gradient ablation、状态恢复、原生接口的 CPU 环境替身、旧 CLI 兼容和忙 GPU 拒绝。**环境替身不是 MuJoCo-Warp 资格验证。**

以下 H1 命令及后续 H8/H32/消融已在固定输入上执行；新运行仍须重新核验空闲 GPU 和 checkpoint/bank SHA，从独立 checkout 使用已固定的 Isaac Lab 环境运行；不更新执行中的 checkout。正式队列运行期间，只在独立空闲 GPU 上做有明确预算的小规模诊断，并记录 CPU/磁盘资源与重叠时段；不得占用正式队列的 GPU 或中断其进程。`CHECKPOINT` 必须为旧 bank 的同一个 SuperMimic collector checkpoint。

```bash
python -m experiments.ablation.fidelity --probe-space policy \
  --checkpoint "$CHECKPOINT" --bank "$BANK" \
  --sample-indices 0 1 --gradient-horizon 1 --probe-seed 9001 \
  --directions 2 --epsilons .001 .003 .01 --repeats 3 \
  --gpu "$GPU" --gpu-uuid "$GPU_UUID" --max-seconds 600 \
  --output logs/ablation/policy-gradient-h1-v1
```

H1 验收后在新的输出目录运行 H8/H32，再分别增加 `--no-critic`、`--detach-action-reward`；四种消融组合不得混入同一目录。默认保留训练 AMP/TF32 设置；`--precision fp32` 关闭 TF32/AMP，用于单独的数值敏感性检查，不能和训练精度结果混合。

默认一次为 34 个物理 rollout（3 基线 + 5 方向×3尺度×正负 + 1 实际 Adam），两个起点的 H1/8/32 分别为 68/544/2176 个物理 transition，另加 prefix 重建。默认预算600秒（0.1667 GPU-hours）是资格上限目标，不是预计耗时；初始化和单个操作可能越界，检查在捕获前和每个 rollout 前执行，真实分配时间保留。初始保守预留16 GiB空闲显存、8 GiB磁盘，均不是已测峰值。正式预算需首次原生运行后根据实测修订，不自动放大批次。

产物：`gradients.pt`、`policy-bank.pt`、`direction-rollouts.pt`、`directions.csv`、`summary.json`、`probe.json`。原 checkpoint、bank 和运行中的训练进程保持不变。失败也保存状态、费用及已有证据。

## 剩余工作与可得结论

1. H1/8/32 原生资格和两种消融已完成。扩展独立随机方向、held-out 状态和接触分层，检查小信号的精度/尺度敏感性。
2. 复用不同 WM 阶段做同 Policy/任务/起点对照，保持前向误差与参数梯度指标独立。
3. 完成上述验收后再加入 PPO 梯度采样和等预算更新。随机 Gaussian PPO 的期望目标与确定性 Policy 的 BPTT 目标不同，必须先统一动作随机性/目标，或把比较明确限制为相同物理评分下的更新有效性；不能把两种目标的梯度夹角直接叫作“准确率”。
4. 正式多 seed 统计和最终归因仍未完成。

目前能明确回答的是：**更好的最终 Policy Performance，本身不能证明 WM Policy Gradient 更准确。** 准确率需要匹配目标的局部数值参考；更新有效性需要实际步长、裁剪和历史优化器状态下的物理改善。真实物理数据已显示代理改善与物理改善可以分离，但范围有限；与 PPO 的匹配目标对照尚未完成。
