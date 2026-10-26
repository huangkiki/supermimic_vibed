# Policy Gradient：准确性与更新有效性

2026-10-26 补充：[独立随机方向与采样收敛实验](estimator-convergence-results.md)已完成。8 个预选方向、N=16/64/256 的结果支持 WM 内 BPTT 的采样优势；主尺度物理参考仅1/8通过筛查，三次额外复评确认大响应可复现但强依赖尺度，因此仍未得到普遍的物理梯度准确率排名。

**SuperMimic 的物理表现优于 PPO，本身不意味着 WM 的完整 Policy Gradient 更准确。** 这两个结论需要不同证据：准确性要比较同一个目标、同一点上的局部导数；有效性要检查选定步长、裁剪和优化器产生的更新是否改善物理 Return。当前单 checkpoint 低成本实验已经完成两条测量链路，支持有限条件下的更新收益，不能给出全参数空间或跨任务的准确率排名。

## 对当前七项要求的验收

| 要求 | 实现与直接证据 | 解释边界 |
|---|---|---|
| 原训练 H32 闭环、Dense Reward、Soft Termination、Critic Bootstrap | `policy_fidelity.py::training_actor_source`直接抽取本仓库`policy_update()` actor前缀；CPU测试运行完整原函数，H1/8/32 loss和Adam位移逐值相同；原生H32实际完成 | Eager执行，未证明与compiled CUDA逐值等价；不改原训练算法 |
| Policy Parameter Gradient入口 | `fidelity --probe-space policy`，9,116,477维全部Actor参数，非动作序列梯度 | 旧Action Gradient入口保留、分别解释 |
| 固定Policy/WM/Critic/replay起点/RNG | 同一checkpoint与bank SHA，保存raw history；每次恢复状态、参数、normalizer与RNG；检查物理初态漂移≤1e-5 | 既有bank重放prefix恢复history，不冒充过去某个训练minibatch的精确重放 |
| 正负参数扰动物理闭环与两个目标 | 三个ε、共同随机数、MuJoCo-Warp完整H步rollout；分别保存soft+bootstrap代理目标和硬终止Return | 中央差分只是数值参考，不是解析真值；硬Return不是代理目标的导数 |
| H1/8/32、符号/收益/噪声、两种独立消融 | [原生15组报告](policy-gradient-native-v1.md)覆盖三个H和2×2消融；[H32 FP32补充](policy-gradient-components.md)补齐精度控制 | Critic关闭改变当前目标；动作stop-gradient改变backward，不改变前向奖励；原两状态条件各两个随机方向；新增16状态接触批次各8随机方向，按尺度报告，仍不推断全局相关性 |
| 复用checkpoint、不重训WM、保留完整梯度和Adam | 原始loss/ascent、norm-clipped、value-clipped、optimizer前后state、实际Adam位移、全部物理rollout均保存；CPU从大张量复核 | 历史Adam moments仍含过去Critic影响，不能称为从头无Critic训练 |
| 同策略同目标PPO采样与相同更新预算 | [匹配目标实验](matched-policy-results.md)：相同mean网络、固定σ=.1，原actor路径BPTT与θ_old处PPO score gradient；16梯度noise tapes、16独立物理noise tapes、相同L2步长 | 此辅助比较是首次PPO梯度，λ=1、无advantage标准化/entropy，不等于完整多epoch PPO训练；noise tapes不是训练seed |

源码、CPU单元测试、原生数据和最终解释相互对应。参数梯度从未穿过MuJoCo-Warp反传；物理目标的参考来自状态恢复后的正负数值实验。

## 实际观察能够说明什么

确定性原路径中，H1/H8的真实历史Adam更新改善了soft+bootstrap代理目标，却降低硬终止物理Return；H32在这两个起点上同时改善两者。Critic消融、直接动作梯度消融与原始/裁剪/Adam方向分别保留，不能把一个因素的变化全归因于WM动力学梯度。

同目标随机对照中，H32沿BPTT平均梯度做L2=.01更新，硬物理Return变化为 **+0.036755 ± 0.001523 SE**；PPO为 **−0.001321 ± 0.001187 SE**。BPTT方向上的匹配代理方向导数预测为11.715814，物理中央差分参考为11.952429 ± 0.235830 SE；它只支持这个选择方向的局部吻合。

PPO均值梯度的采样SE L2为214.149833，均值L2为235.628121，当前16个梯度noise tapes不足以把它当作收敛的期望梯度。不能把PPO估计不稳定与WM模型偏差混为一谈；也不能从两个选择方向的夹角推断9,116,477维梯度的整体准确性。上述SE描述固定起点上的噪声采样，不是训练seed不确定性或全部数值误差。

统一随机策略与WM期望目标后，在可交换微分与期望等条件下，pathwise与score估计针对同一个期望梯度。当前对照显示有限采样下的方差和更新效果不同；不支持“BPTT理论期望梯度天然比PPO更正确”。这个WM期望梯度本身还可能偏离物理目标梯度，详见[同目标协议](matched-policy-estimators.md)。

新增[两批各8状态的接触诊断](policy-gradient-contacts.md)进一步区分了两个问题：实际Adam平均硬回报改善 +0.006009 / +0.018797，但在 ε=.01 的8个随机方向上，匹配代理参考仅1/8与0/8可解析；ε=.001/.003则为2/8与4/8，结果依赖尺度。过去stable组7/8在未来H32发生切换，因此没有得出接触稳定性与梯度准确性的因果关系。该批使用明确标记的派生状态库，不与此前原末端库混合。

## 已验证结果、观察与未证实推测

- **已验证**：原actor公式/裁剪/历史Adam复用；固定快照和共同随机数；真实物理参数扰动；目标分离；保存梯度与方向点积复核；H32完整FP32消融；匹配随机目标的两种梯度估计器及物理更新链路。
- **观察性结果**：当前checkpoint和两个run起点上，H32 BPTT的等步长更新更有效，采样波动更小；代理收益与硬物理收益可以分离。这条BPTT/PPO比较仍只覆盖两个run起点；新增[16状态接触诊断](policy-gradient-contacts.md)覆盖walk/run/sprint，未重新做PPO对照。
- **未证实**：全向量梯度准确性排序、PPO充分采样后的方向、跨状态/接触/WM阶段/训练seed的稳定性、同等最终质量下的完整训练速度优势。低State Prediction MSE仍不能代替任何Policy Gradient准确性指标。

本次固定checkpoint、单seed、低成本参数梯度验证与原始三方法长程实验分别验收。正式三训练seed/40段、WM吞吐与优化机制的端到端归因仍需完整队列结果；本报告不将这些后续研究写成已完成。
