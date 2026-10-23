# 已验证的阶段结果（更新至2026-10-24）

本PR已完成原生Smoke、100迭代单seed Pilot、18段4096规模资源资格、64状态探索性梯度诊断及开发seed0的六条件4000轮矩阵。尚无完整三seed或40段正式结果；正式seed1已启动，seeds2/3串行等待，40段资源资格另行测量。

训练/Pilot固定提交555a051；初版数值探针e410334；配对full/state_only探针8d02575；长程运行器30739d2。原supermimic目录未修改。先前26项CPU回归测试及对应CI通过；新增独立Actor学习率后29项原生CPU测试通过，原生状态/FK/reset资格通过。

## 统一短程物理评估

同一A100-PCIE-40GB GPU、seed0、18段walk/run/sprint、256环境、H32、100迭代。7次训练（含pretrain）+30次物理复评，540个clip记录；实际campaign0.75351 GPU-hours。所有条件跟踪失败率100%，不能排名。最终归一化物理return约0.00146–0.00157；BPTT冷编译占短程成本很大比例。

![Returns](../../results/ablation/pilot-v1/return-compute.png)
![Costs](../../results/ablation/pilot-v1/compute-breakdown.png)

[评估与完整成本CSV](../../results/ablation/pilot-v1/evaluations.csv)。冻结方法曲线各自计入全部预训练；实际campaign只执行一次共享pretrain。中间成本点为重建快照，最终点使用完整进程分配时间。单seed无标准差带。

## 4000 迭代探索：PPO 基线尚未合格

2026-10-24 00:28快照：PPO-Sim、在线PPO-WM和SuperMimic均已完成4000轮训练及各5次物理评估；冻结条件继续执行。硬件、18段动作和评分协议一致。

| 迭代 | 原PPO-Sim Return | 原在线PPO-WM Return | BPTT Return | BPTT失败数/18 |
|---:|---:|---:|---:|---:|
| 0 | 0.001492 | 0.001492 | 0.001492 | 18 |
| 1000 | 0.003371 | 0.001674 | 0.376017 | 8 |
| 2000 | 0.009728 | 0.001717 | 0.590711 | 5 |
| 3000 | 0.005257 | 0.002987 | 0.751716 | 3 |
| 4000 | 0.003055 | 0.001697 | 0.803472 | 2 |

两个原PPO条件每个点均18/18提前失败，不能把这些数值解释为算法排名。PPO-Sim 99.75% 迭代触发 KL 提前停止，通常只有少数 actor minibatch 更新。下一 minibatch 的停止不撤销上一步过大的更新。

![探索长跑的部分物理曲线，PPO尚未合格](../../results/ablation/formal-seed0-partial/return-compute.png)

[完整成本与评估 CSV](../../results/ablation/formal-seed0-partial/evaluations.csv)。曲线只包含已完成物理评估点；尚未完成的checkpoint不补点。BPTT已有明显物理改善，但PPO基线未合格，不能直接给出优化器因果收益。单seed、旧协议每点一次复评，无标准差带。

SuperMimic端点训练进程5123.91秒、五次物理评估进程合计1652.88秒，总计1.88244 GPU-hours；训练物理步数41,222,144，imagined步数458,752,000，评估物理步数973,548另列。最终checkpoint SHA256为`54d0854c1b4a8af9770b0c6f11c29d6ad8d0c9e6f870913e81234ac9caf0cbac`。中间点是训练快照加累计复评的回溯成本，实际评估在完整训练后执行；图轴据此标为accounted time，而非分数实际可用时间。

在同一真实 PPO-Sim 批次与 Adam 状态上，仅将 actor 学习率 2e-4 改为 5e-6，更新后的整批精确 KL 从 0.832 降到 0.013，20 次 minibatch 更新完成；归一化更新单独产生的 KL 约 1e-5。该诊断支持步长失配。后续两候选均完成1000轮和9次物理复评：lr=5e-6达到Return=0.013086、18/18失败；lr=2e-5达到0.252798、12/18失败（3次复评SD=2.198e-6），已出现有效学习。正式长程验证选用2e-5，开发seed0不计入正式seeds1/2/3。代码新增独立 `actor_lr` 配置，缺省仍继承原 `lr`。[完整诊断、原始数据与复现命令](ppo-qualification.md)。

## 开发seed0的六条件矩阵完成

2026-10-24 02:46完成原生验收：7次训练（包括一次共享WM预训练）、30次物理评估，共37个成功子进程，实际6.678393 GPU-hours，未超过原14 GPU-hours上限。原资源门禁中断记录保留；恢复使用原30739d2训练源码和已存在checkpoint，没有重训或重置预算。

| 第4000轮方法 | 物理Return | 失败数/18 | 单独运行的训练及评估GPU-hours |
|---|---:|---:|---:|
| PPO-Sim（原参数） | 0.003055 | 18 | 1.674874 |
| PPO-WM online（原参数） | 0.001697 | 18 | 0.980082 |
| SuperMimic | 0.803472 | 2 | 1.882443 |
| PPO-WM frozen hard（原参数） | 0.000485 | 18 | 0.607807 |
| PPO-WM frozen soft（原参数） | -0.011339 | 18 | 0.565257 |
| BPTT-WM frozen | 0.001907 | 18 | 1.184420 |

每个冻结条件的成本都独立计入完整WM预训练；实际campaign只预训练一次，因此表中方法成本之和不等于campaign账本。每个checkpoint仅一次物理复评，不能报告训练seed标准差；4000轮也不等于相同计算预算。

![完整六条件的开发seed0曲线](../../results/ablation/formal-seed0-complete/return-compute.png)

[30个评估点与成本CSV](../../results/ablation/formal-seed0-complete/evaluations.csv) · [完整进程账本](../../results/ablation/formal-seed0-complete/campaign.json) · [原生验收](../../results/ablation/formal-seed0-complete/validation.json)。旧三条件部分结果保留为历史快照。

冻结BPTT使用500轮预训练的固定WM/replay，训练过程中没有新增模型拟合和物理采集。即使采用BPTT，它也未得到有效物理跟踪；固定模型及数据覆盖可能限制效果，但本组实验不能区分模型成熟度、分布偏移和优化设置的影响。原PPO参数未合格，不能把表中差距直接归因于BPTT收益或给出算法加速比。正式seeds1/2/3沿用已经预登记的设置和预算，独立复验，不以该开发结果调整正式批次。

截至02:52，正式seed1在GPU0完成500轮WM预训练，PPO-Sim训练进程已启动；seeds2/3等待。正式训练源码固定3526437，与分析代码分离，旧seed0不计入正式均值/标准差。此状态不表示seed1已有最终策略成绩。

## 奖励项对梯度诊断的混杂

同一64状态/动作库、4个WM阶段、4随机方向/状态、4扰动尺度；matched soft与硬物理目标分列。FD是尺度依赖的数值参考。起点漂移最大1.19e-7；未通过噪声/尺度稳定性筛选的方向保留为unresolved。

checkpoint100、ε=.03、双方resolved交集：H1 n=246/256，full/state-only符号一致率1.000/.614，Pearson .997/.250；H8 n=139，一致率.842/.576；H32 n=134，一致率.761/.470。完整奖励包含可直接解析的动作平滑项，高一致率不能独立证明动力学梯度可靠。

![Paired objectives](../../results/ablation/paired-components/paired-objectives.png)

[所有阶段/尺度/接触组的配对统计](../../results/ablation/paired-components/paired-gradient-metrics.csv)。state-only是诊断后新增的辅助控制，不修改训练/主评估reward。采样库由参考PD切换到欠训练collector，首步动作跳变平均L2=9.45，放大了动作项；不能把结果外推至成熟策略。

body接触定义下H1/8/32分别22/14/3个稳定起点；geom接触斑块层级没有稳定组，两个定义都保存。H32稳定样本太少，不做显著性结论。所有方向复用64个状态，不能当独立训练seed。

![State-only diagnostic](../../results/ablation/paired-components/gradient-components-state_only-h32/prediction-gradient.png)

H32、ε=.03沿full奖励WM梯度微步的真实硬物理累计回报平均改善+.0010034；这不是完整motion归一化return，也不是policy参数更新后的学习效果。state-only辅助分数改善+.0003060另报。

## 后续与资源

4096环境实测峰值PPO-Sim8450MiB、在线PPO-WM12422MiB、BPTT23638MiB。BPTT补测最后10次更新1.135–1.203秒，cold编译成本另保留。旧的单seed六条件4000迭代估算为8.52–10.77 GPU-hours；它尚未计入充分Actor更新与每点三次复评，不能用于下一批正式运行。冻结吞吐与评估时长含明确规划假设，4000迭代不保证收敛。

原预算中评估时长按30–300秒估算；BPTT第1000轮的完整物理评估实际396.49秒，超过旧估算上界。下一批正式运行必须根据成功跟踪后的实测时长重新估算，不能复用该旧评估假设。当前旧矩阵14 GPU-hours硬上限仍生效。新PPO候选复评实际395–407秒，在线PPO-WM的500轮预热+32轮完整更新资格共0.161523 GPU-hours；据此重新估算正式seeds1/2/3共94.17 GPU-hours，分seed上限32、合计96，显存预留27.7 GiB。详见[PPO资格与预算](ppo-qualification.md#两候选完成与正式参数冻结)。

长程seed0完成后先检查有效学习和成本，再扩展至少3seed与40段；已训练collector的首组固定库梯度分析见下节，多seed结论仍待执行。复现见[runbook](runbook.md)，因果边界/剩余算法差异见[protocol](protocol.md)，固定环境见[environment](environment.md)。完整逐步原始日志、接触baseline、数据manifest和实验报告交付于本任务outputs；大模型checkpoint保留在远端专用实验目录，另附SHA/路径清单。没有合并本PR，也没有宣称最终研究完成。

## 已训练策略的固定库：前向误差降低，长时域梯度仍需独立检验

2026-10-24完成六个新探针：固定预先选定的最终4000轮collector（物理Return=0.803472，仍2/18失败），64个参考起点经16步连续policy前缀，probe seed=9001；同一状态/动作/名义物理参数库比较WM第1000/2000/3000/4000轮。full/state_only × H1/8/32，每探针1024条prediction及10240条gradient记录。GPU2共0.188922 GPU-hours，与主训练同主机重叠，成本另列；不是吞吐排名数据。

执行源码6dd6fd5，bank SHA256=`a79f08f281b288e8f8001aa994fc127789c080e42b67fa809dce1cbeb48d3abb`。全部探针内容身份与记录检查通过；初态最大漂移1.192e-7，首动作重放最大误差9.537e-7。边界/后续动作变化均值0.748/0.626，前缀一轮采集成功；这是参考起点的条件采样，仍不是平稳on-policy分布，也不是未见motion泛化。

| WM迭代 | 1-step MSE | 8-step MSE | 16-step MSE | 32-step MSE | 同checkpoint物理Return |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.007002 | 0.006863 | 0.020342 | 0.115979 | 0.376017 |
| 2000 | 0.000829 | 0.001645 | 0.003458 | 0.013098 | 0.590711 |
| 3000 | 0.000513 | 0.000931 | 0.001627 | 0.004460 | 0.751716 |
| 4000 | 0.000442 | 0.000589 | 0.000730 | 0.003024 | 0.803472 |

MSE来自full-H32的同一固定库预测，采用协议中的固定分量尺度，不是论文原始MSE。四阶段的MSE与策略表现共同改善，但WM、policy和训练数据同时成熟，这个观察不能证明MSE降低导致策略改善。每checkpoint旧协议仅一次物理评估，无评估标准差；四个阶段不是四个独立训练seed。

![同库预测误差与物理表现](../../results/ablation/trained-policy-fidelity-seed0-v1/gradient-components-full-h32/prediction-performance.png)

第4000轮、ε=.03、matched soft目标，full/state_only使用双方均resolved的同一随机方向交集：

| 梯度时域 | 交集方向数/256 | full符号一致率 | state-only符号一致率 | full Pearson | state-only Pearson |
|---:|---:|---:|---:|---:|---:|
| 1 | 249 | 0.936 | 0.900 | 0.966 | 0.941 |
| 8 | 199 | 0.864 | 0.874 | 0.868 | 0.869 |
| 32 | 140 | 0.664 | 0.664 | 0.094 | 0.295 |

![四扰动尺度下的配对方向统计](../../results/ablation/trained-policy-fidelity-seed0-v1/paired-objectives.png)

短时域的一致性明显高于H32；H32的结果对扰动尺度也敏感，完整图保留四个尺度及各自分母。不能把未resolved方向判作错误，或把筛选后的140个方向称作独立样本。这组结果说明前向误差与方向质量需要分开测量；不证明BPTT训练无效，因为此探针是有限时域动作序列方向、无critic，和训练中的policy参数梯度并不相同。

H32、full目标、ε=.03，沿各WM单位梯度方向微步的真实硬目标累计回报改善均值（64状态、含负改善）依次为+0.00000304、+0.00023420、+0.00096606、+0.00077143。第4000轮同尺度随机单位方向为−0.00004468；其四尺度梯度微步均值为+0.00007150/+0.00003824/+0.00077143/+0.00209790。3000→4000的MSE下降未带来该局部微步指标的单调改善，但没有独立训练seed置信区间，不能声称差异显著。这些值不是全motion归一化Return，也不是重新训练后的policy improvement。

接触分组的状态数：H1稳定/切换=35/29，H8=17/47，H32=3/61。H8、ε=.03、state-only在稳定/切换组的符号一致率为0.922/0.852，双方resolved方向64/135；仅是该条件下的观察。H32稳定组只有3状态，不能据此做可靠的接触因果结论；geom原始接触记录也保留。

[全部阶段/尺度/接触组CSV](../../results/ablation/trained-policy-fidelity-seed0-v1/paired-gradient-metrics.csv) · [MSE与物理Return CSV](../../results/ablation/trained-policy-fidelity-seed0-v1/gradient-components-full-h32/prediction-performance.csv) · [校验、来源与成本收据](../../results/ablation/trained-policy-fidelity-seed0-v1/analysis-receipt.json)。


## 连续策略前缀与状态库内容验收

新增policy roll-in保持同一collector、previous action和观测历史跨越探针边界，reference_pd保留原切换控制。四个原生探针（两模式×full/state_only）均完成，18起点/模式、checkpoint100、H1、1个随机方向、eps=.01/.03，合计0.0338113 GPU-hours；每探针72条prediction、144条gradient记录。首动作从保存的状态/历史重放，最大误差2.384e-7；物理初态最大漂移1.192e-7。44项CPU测试通过。

参考PD模式的边界动作L2变化均值9.8799、后续变化均值0.3391；policy模式分别0.3210/0.2359。两组存活筛选分别重放3/2轮，状态分布不同，不能把这两个均值之差当作同状态因果效应或策略性能改善；collector仍欠训练。这组证据只验证连续历史、记录、内容身份和原生FD链路，不替代成熟策略实验。运行在GPU2，与主训练同主机重叠，成本单列。

配对检查的旧路径判断缺陷已由回归测试复现：同一路径但不同内容未被拒绝、两个未标识新库被当作配对。修复后新记录必须匹配bank SHA256、前缀和物理GPU UUID；旧路径证据显式标记legacy_path_only，不补造历史哈希。

![状态库动作连续性](../../results/ablation/policy-bank-smoke-v1/bank-continuity.png)

[验收摘要](../../results/ablation/policy-bank-smoke-v1/qualification.json) · [逐状态动作变化](../../results/ablation/policy-bank-smoke-v1/bank-diagnostics.csv) · [同库配对梯度统计](../../results/ablation/policy-bank-smoke-v1/paired-gradient-metrics.csv)。原始探针收据、状态库、CSV和日志保留在交付数据包。解包后可重绘：

```bash
python results/ablation/policy-bank-smoke-v1/reproduce.py /path/to/policy-bank-smoke-v1
```

## Prediction Error与物理表现的重复评估汇总

修复了绘图入口按checkpoint覆盖复评、只取最后一条Return的问题。实际Matplotlib入口回归测试中，三次Return为0.2/0.4/0.9，旧输出0.9，修复后均值0.5、样本标准差0.360555，缺失第三次复评时拒绝绘图；46项CPU测试通过。该修复不改变已有单次评估的物理分数。

已有欠训练Pilot四个WM checkpoint重新生成1/8/16/32步误差与物理Return的关联图。每点只有一次旧协议评估，标准差明确为空；没有补造复评或训练seed。图中物理表现仍接近失败下限，不能用四个相关checkpoint推断MSE的因果作用。

![四时域误差与物理表现](../../results/ablation/prediction-performance-v2/prediction-performance.png)

[逐点CSV](../../results/ablation/prediction-performance-v2/prediction-performance.csv) · [分析输入收据](../../results/ablation/prediction-performance-v2/analysis-receipt.json)。

## 全部40段的原生 Smoke（2026-10-24）

独立版本3526437、A100 GPU2、seed0完成全部40段输入与六条件链路检查：pretrain 1轮，其余各3轮，共7个训练进程、12次完整物理评估、480条逐clip记录，总0.1975698225 GPU-hours。原生验收核对Actor更新、冻结WM不变、bootstrap成本、checkpoint身份与动作覆盖。所有策略均40/40失败；这仅证明全量动作链路可执行，不是40段正式训练，也不进入18段性能或吞吐比较。

[CSV](../../results/ablation/all40-smoke-v1/evaluations.csv) · [Return/Compute图](../../results/ablation/all40-smoke-v1/return-compute.png) · [验收记录](../../results/ablation/all40-smoke-v1/validation.json)。正式全量扩展仍需4096环境的独立资源测量与至少3个训练seed。
