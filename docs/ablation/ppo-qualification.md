# PPO 基线资格：先排除更新尺度失配

当前单 seed 长跑仍属于探索实验。PPO-Sim / 在线 PPO-WM 已各执行 4000 次迭代，但所有 18 段动作仍提前失败，不能据此宣布 BPTT 优于 PPO。这个文件记录可复核的失败原因诊断和后续资格标准。

## 已复现的更新问题

从 PPO-Sim 第 1000 次迭代 checkpoint 采集真实物理 rollout：512 环境 × 32 步 = 16384 样本，固定 seed=9021。之后不再重新采样，所有候选复用同一 actor、critic、Adam 状态、优势/价值目标和小批次顺序，只改变 Actor 学习率。每个候选深复制 Adam 状态，避免候选之间互相污染。

| Actor 学习率 | 实际 actor 更新数 / 20 | 优化后精确 KL 均值 | 归一化更新单独造成的 KL | 包含归一化的总 KL |
|---|---:|---:|---:|---:|
| 2e-4 | 1 | 0.831662 | 9.82e-6 | 0.830999 |
| 2e-5 | 3 | 0.022878 | 9.73e-6 | 0.023027 |
| 5e-6 | 20 | 0.013033 | 9.90e-6 | 0.013035 |
| 1e-6 | 20 | 0.004030 | 9.87e-6 | 0.004027 |

KL 为整批状态上对角高斯的 `KL(old || new)`，不是学习曲线里 minibatch 的近似 KL 最大值。三个 KL 量不能直接相加。更新前 eager/compiled log-prob 差异对应的近似 KL 为 4.00e-7。重新运行 2e-4 基线得到相同结果；5e-6 候选通过平均 KL≤0.02 检查。

原 PPO 的 early-stop 在**下一次** minibatch 更新前检查 KL，不会撤销上一步。因此阈值不等价于严格信赖域保证；这不是把 PPO 宣称为有硬约束后再判定其实现错误。此处验证的是现有学习率、动作标准差和优化状态组合的更新过大现象。

[原始 replay JSON](../../results/ablation/ppo-diagnostic-v1/replay.json) · [学习率对照 CSV](../../results/ablation/ppo-diagnostic-v1/learning-rate-sweep.csv)。checkpoint SHA256=`466d8cd9448902bd2231e470ad35556f29ae988414f44e9ffd9ca4cf3b240059`，batch SHA256=`660f04dff3eacffc3e0c0549a49ae42f8e049035a0fcc2757856f0fd42153814`。大文件留在远端专用实验目录，不提交 Git。

## 代码变化与复现

`actor_lr` 是可选配置，缺省/`null` 时继承 `lr`，旧配置行为不变。Actor 使用独立基值和同一余弦衰减，WM/Critic 继续使用 `lr`。每次迭代的 CSV 同时记录两者；checkpoint 保存配置和实际 optimizer param groups。没有把旧长跑中途换成新参数，也没有改变原 `supermimic/`。

在已经通过原生环境资格的仓库根目录运行。`CHECKPOINT` 指向上面的 PPO-Sim checkpoint，`BATCH` 可指向已有捕获文件；输出路径必须是新目录。

```bash
python -m experiments.ablation.diagnose_ppo capture \
  --checkpoint "$CHECKPOINT" --num-envs 512 --seed 9021 \
  --output logs/ppo-diagnostic/capture
python -m experiments.ablation.diagnose_ppo replay \
  --batch logs/ppo-diagnostic/capture/batch.pt \
  --checkpoint "$CHECKPOINT" \
  --learning-rates 0.0002 0.00002 0.000005 0.000001 \
  --output logs/ppo-diagnostic/sweep
# 原始固定批次上预期失败；不能将这个非零退出删除或记作成功。
python -m experiments.ablation.diagnose_ppo replay \
  --batch "$BATCH" --checkpoint "$CHECKPOINT" \
  --learning-rates 0.0002 --assert-max-kl 0.02 \
  --output logs/ppo-diagnostic/red
# 原始固定批次上通过；只证明这个更新的尺度合适。
python -m experiments.ablation.diagnose_ppo replay \
  --batch "$BATCH" --checkpoint "$CHECKPOINT" \
  --learning-rates 0.000005 --assert-max-kl 0.02 \
  --output logs/ppo-diagnostic/green
```

Capture 使用真实新仿真数据，不等于恢复训练过程中同一 simulator/RNG 状态；其目标是获得可重复重放的诊断批次。迁移 checkpoint 路径可用 `--checkpoint`，SHA 不匹配会拒绝运行。GPU0 上的诊断与 GPU1 长跑有同主机重叠，成本单独记账，不纳入正式吞吐比较。

## 原生短程与后续资格

`qualification-ppo_sim.json` 保留正式配置的 4096 环境、H32、seed0、std0.1 和 critic/model 学习率，只指定 actor_lr=5e-6，32 次迭代、1200 秒内预算。短程检查的目的：实际调度器不覆盖独立学习率、所有更新和独立物理评估有限、测量充分 minibatch 更新下的耗时和显存。它不等于 4000 次学习实验的修复验证。

原生验证已完成：32/32 迭代均完成 20 次 actor 更新，checkpoint 中 actor/critic 学习率比值保持 0.025；最后 10 轮用时 1.975–2.010 秒，Torch 峰值显存 4303 MiB。物理归一化 Return=0.001503，18/18 仍提前失败。录制轨迹的 18 个终止帧与逐 clip 步数一致，终止后姿态保持检查通过。这些检查不能替代长程学习验证。

新长跑前，需要根据该短程重新估算成本。之后用有界训练比较候选学习率和必要的探索标准差，保留调参成本及失败；用物理 Return、存活步数和失败率判断是否学习，KL 只作为诊断。候选确定后再运行至少 3 个训练 seed、40 段扩展及成熟 WM 的梯度实验。所有调参选择与最终统计中的 seed 复用情况均需披露，不能把调参曲线充当独立复现。


## 物理复评的重复性限制

同一个 PPO-Sim iteration4000 checkpoint，在同一 GPU0 上启动新进程、同一 seed=0、确定性 mean 动作与物理参数，实测如下。录制均使用同一版本的评估代码，依次进行一次 on、两次 off、一次 on。

| 录制 | 物理归一化 Return | 活跃仿真步数 | 失败 |
|---|---:|---:|---:|
| on1 | 0.003416 | 2522 | 18/18 |
| off1 | 0.002815 | 2097 | 18/18 |
| off2 | 0.003232 | 2427 | 18/18 |
| on2 | 0.002872 | 2085 | 18/18 |

关闭录制的两次结果也不同，因此不能将跨 GPU 或单次 on/off 差异全部归因于录制。样本仅四次，尚不能证明录制对数值轨迹没有影响，也不能定位到某个 solver/kernel。静态实现没有改动作、物理参数或评分；这些分数是单独的重复性审计，不替换主曲线的原始评估。

[逐次 CSV](../../results/ablation/ppo-diagnostic-v1/evaluation-repeats.csv)。后续正式结果须把评估重复的波动与训练 seed 间的标准差分开；增加训练 seed 不能自动消除单个 checkpoint 的数值敏感性。当前四次均失败的判定一致，细小 Return 差异不作算法收益证据。
