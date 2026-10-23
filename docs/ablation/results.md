# 已验证的阶段结果（2026-10-23）

本PR已完成原生Smoke、100迭代单seed Pilot、4096规模资源资格和64状态探索性梯度诊断。尚无收敛后的三seed或40段结果；单seed4000迭代矩阵正在独立工作目录执行，14 GPU-hours上限。

训练/Pilot固定提交555a051；初版数值探针e410334；配对full/state_only探针8d02575；长程运行器30739d2。原supermimic目录未修改。26项CPU回归测试及对应CI通过，原生状态/FK/reset资格通过。

## 统一短程物理评估

同一A100-PCIE-40GB GPU、seed0、18段walk/run/sprint、256环境、H32、100迭代。7次训练（含pretrain）+30次物理复评，540个clip记录；实际campaign0.75351 GPU-hours。所有条件跟踪失败率100%，不能排名。最终归一化物理return约0.00146–0.00157；BPTT冷编译占短程成本很大比例。

![Returns](../../results/ablation/pilot-v1/return-compute.png)
![Costs](../../results/ablation/pilot-v1/compute-breakdown.png)

[评估与完整成本CSV](../../results/ablation/pilot-v1/evaluations.csv)。冻结方法曲线各自计入全部预训练；实际campaign只执行一次共享pretrain。中间成本点为重建快照，最终点使用完整进程分配时间。单seed无标准差带。

## 奖励项对梯度诊断的混杂

同一64状态/动作库、4个WM阶段、4随机方向/状态、4扰动尺度；matched soft与硬物理目标分列。FD是尺度依赖的数值参考。起点漂移最大1.19e-7；未通过噪声/尺度稳定性筛选的方向保留为unresolved。

checkpoint100、ε=.03、双方resolved交集：H1 n=246/256，full/state-only符号一致率1.000/.614，Pearson .997/.250；H8 n=139，一致率.842/.576；H32 n=134，一致率.761/.470。完整奖励包含可直接解析的动作平滑项，高一致率不能独立证明动力学梯度可靠。

![Paired objectives](../../results/ablation/paired-components/paired-objectives.png)

[所有阶段/尺度/接触组的配对统计](../../results/ablation/paired-components/paired-gradient-metrics.csv)。state-only是诊断后新增的辅助控制，不修改训练/主评估reward。采样库由参考PD切换到欠训练collector，首步动作跳变平均L2=9.45，放大了动作项；不能把结果外推至成熟策略。

body接触定义下H1/8/32分别22/14/3个稳定起点；geom接触斑块层级没有稳定组，两个定义都保存。H32稳定样本太少，不做显著性结论。所有方向复用64个状态，不能当独立训练seed。

![State-only diagnostic](../../results/ablation/paired-components/gradient-components-state_only-h32/prediction-gradient.png)

H32、ε=.03沿full奖励WM梯度微步的真实硬物理累计回报平均改善+.0010034；这不是完整motion归一化return，也不是policy参数更新后的学习效果。state-only辅助分数改善+.0003060另报。

## 后续与资源

4096环境实测峰值PPO-Sim8450MiB、在线PPO-WM12422MiB、BPTT23638MiB。BPTT补测最后10次更新1.135–1.203秒，cold编译成本另保留。单seed六条件4000迭代预计8.52–10.77 GPU-hours，3seed25.55–32.30；加入25%余量分别13.46/40.38。冻结吞吐与评估时长含明确规划假设，4000迭代不保证收敛。

长程seed0完成后先检查有效学习和成本，再扩展至少3seed与40段；成熟模型的固定库梯度分析仍待执行。复现见[runbook](runbook.md)，因果边界/剩余算法差异见[protocol](protocol.md)，固定环境见[environment](environment.md)。完整逐步原始日志、接触baseline、数据manifest和实验报告交付于本任务outputs；大模型checkpoint保留在远端专用实验目录，另附SHA/路径清单。没有合并本PR，也没有宣称最终研究完成。
