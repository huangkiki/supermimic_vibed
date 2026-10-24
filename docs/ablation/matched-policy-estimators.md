# 同策略、同目标的 BPTT / PPO 梯度估计

这是完整确定性 Policy 梯度实验之后的辅助对照。两种估计器共用同一个 Policy mean、固定 Gaussian σ、WM、Critic、起点及完整软终止目标。原确定性训练与已有探针不变；**不能将辅助随机目标的结果改写为原确定性梯度的准确率。**

当前先完成 CPU 数学与实现验收；真实机器人结果须以成功的原生收据为准。PPO 使用 [PPO 原论文](https://arxiv.org/abs/1707.06347) 的 likelihood-ratio clipped surrogate，在 θ=θ_old 计算首次梯度，此处 ratio=1、clipping 尚不生效。本实验不是完整多 epoch PPO 训练。

## 匹配条件

两个估计器都令 `a_t=μ_θ(history_t,goal_t)+σz_t`，σ=.1 固定，不引入可学习的 log_std，参数维度完全相同。没有额外 action squashing/clipping。原有训练中的随机 PPO 学习 log_std；本辅助对照明确去掉这一结构差异，限制在相同 mean 参数上。

- BPTT：同一个原 `policy_update` actor 前缀，Gaussian 动作使用 reparameterization，全目标反传。
- PPO：相同 seed 再采集同一 WM 轨迹；断开 obs、goal、action、advantage、旧 log_prob 与 survival weight 的图，只对新 log_prob 求导。固定 Critic，不更新任何归一化统计。
- 尾部回报 `G_H=V_H`，`G_t=r_t+γ(1-d_t)G_{t+1}`；无 Critic 时 `G_H=0`。使用 λ=1，无 advantage 标准化，无 entropy term。
- `A_t=G_t-V(s_t)`，value 是动作发生前的冻结基线；目标权重为 `γ^t c_t`。原 PPO loss 平均 H×B 个项，本对照乘 H，恢复原目标按时间求和、按起点求均值的单位。
- 原路径两种 gradient clipping 和历史 Adam 实验保留在独立报告。本对照使用两种**平均原始梯度的单位 L2 方向**和相同 ε，隔离参数步长；没有把沿 BPTT 历史 moments 预条件后的 PPO 方向称为标准 PPO 更新。

## 采样与解释

`--estimator-seeds` 是固定起点上独立动作/观测噪声 tape。每个 seed 两种估计器的动作和目标须逐值相同；保留各自完整向量、脱离计算图的 rollout 和计时。输出 sample variance trace、均值向量、均值标准误 L2、相对方差；这些属于条件于固定 checkpoint/起点的采样方差，**不是多个训练 seed**。

物理差分使用完全独立的 `--reference-seeds`。正负扰动共享对应 tape，但不同 tape 的方向差分保留为独立参考样本。分别报告参考均值、样本 SD、SE，不能用一次差分当期望梯度真值。ε 敏感性/同 tape 数值重复噪声与跨 tape 采样误差分开；数值筛查 resolved 不表示采样不确定性已经消失。软目标导数与硬物理更新收益仍分别解释。

统计量至少需要更多样本与方差收敛检查才能支持稳定结论；一个 mean-gradient cosine 或单次收益不能证明模型梯度准确性。本切片只比较两条估计器方向，没有充分随机方向来计算空间范围的相关性。

## 命令与预算

先跑 CPU 检查（含解析 Gaussian 期望梯度、H8 causal weighting、H1 Critic Bootstrap、固定噪声的闭环中央差分、PPO 无 WM 反传和独立噪声组）：

```bash
python -m unittest discover -s tests -p test_policy_estimators.py -v
```

以下是原生 H1 资格命令。输入 checkpoint、准备好的 raw-history bank 与环境保持[上一批原生实验](policy-gradient-native-v1.md)身份；新输出目录不可覆盖，先检查 GPU/磁盘/CPU，外部 controller 另加硬超时。

```bash
python -m experiments.ablation.fidelity --probe-space policy \
  --checkpoint "$CHECKPOINT" --bank "$PREPARED_BANK" --sample-indices 0 1 \
  --gradient-horizon 1 --precision fp32 --action-std .1 \
  --estimator-seeds 9001 9002 9003 9004 \
  --reference-seeds 19001 19002 19003 19004 \
  --epsilons .001 .003 .01 --repeats 3 \
  --gpu "$GPU" --gpu-uuid "$GPU_UUID" --max-seconds 110 \
  --output logs/ablation/matched-policy-h1-v1
```

该命令计划 54 次物理 rollout（4×13+2）和两种估计器各4×H×2个 imagined transition。后续 H8/H32 需根据资格实测独立审查，不能按示例自动启动长实验。原生结果、完整进程计费与硬超时由外部收据保留；脚本内预算是 cooperative，单个操作可能超时。

产物：每个 seed 的 `estimate-*.pt`，`mean-gradients.pt`，`physical-reference.pt`，`estimator-costs.csv`、`sampling-variance.csv`、`comparison.csv`、`comparison-summary.json` 及原生环境收据。两个方法在同 GPU、同精度、同起点的相同 trajectory 数量下比较；计时包含各自 WM rollout 和梯度计算，但单次微小诊断的 CUDA 热身/顺序效应使其不适合直接报告训练加速比。
