# 固定 Policy 的 WM / 物理梯度实验（预注册 v1）

要回答的问题有两个：降低 WM forward error 是否保证更可靠的 Policy 梯度；SuperMimic 胜过 PPO 是否能归因于梯度更准确。此前同 WM 实验已发现 BPTT 采样方差更小，但物理全参数随机方向只有 1/8 可分辨，因此尚不能作物理准确性排名。

本轮增加实际 PPO-Sim score gradient、五个 WM checkpoint、同动作预测误差及真实物理小步更新收益。延续原始八个全参数方向结果，不用本轮的子空间结果覆盖原结论。

- 固定 seed0 / update4000 Policy、Critic、两条 run 起点及 raw history；WM 分别取 0/1000/2000/3000/4000，连同各自的输入输出 normalizer。检查原始源码、资产、motion、软件环境身份。
- 同一个固定 sigma=0.1 的 Gaussian Policy，H=32、FP32、TF32 关闭、eager。原始 `train.py` actor prefix 给出 soft-survival + frozen-critic bootstrap 目标；物理 hard return 单列。PPO 指 theta_old 处、lambda=1、无 advantage normalization / entropy 的 score estimator，不代表完整多轮 PPO 训练。
- 每起点 16 个环境副本（batch32），gradient seeds 61001..61016，reference seeds 71001..71016。副本/噪声是 Monte Carlo 重复，不是训练种子。所有方法保持同一 batch shape。
- 8 个提前生成的 29 维输出 bias Rademacher 单位方向，PCG64 seed81001；与模型/梯度独立。epsilon=.001/.003/.01，.003 为主尺度。覆盖仅限该子空间；9,116,477 维全参数均值另外用于选定更新方向的有效性测试。
- BPTT / PPO-WM 使用共同噪声及相同目标；PPO-Sim 使用相同随机政策在原生 MuJoCo-Warp rollout，再对 detached observations/actions/targets 求 score gradient。物理 rollout 不参与反传。
- 所有 WM 用同一套物理状态及其实际动作序列计算 open-loop H1/8/16/32 error；固定尺度 [p .3, rotation .4, v 1, w 3.14, q 1, qd 10]，不采用随 checkpoint 改变的训练 normalizer 作为误差单位。按原生 body-contact pair 集合区分 stable_contact、switch_nearby、flight。
- 物理中央差分使用共享噪声、各起点相同 xi、冷 solver reset；3 次 baseline 重复。参考的相邻尺度一致性 25%、信号高于 5 倍 repeat noise / 1e-7、幅值高于 2 MC SE；未解析方向必须保留，有限差分不是绝对真值。报告 per-scale 原始值以及可解析分母。
- 五种 BPTT、五种 PPO-WM、PPO-Sim 共 11 个完整均值方向，以相同参数 L2 步长 .001/.003/.01 在物理环境做配对评测，保留初始 action RMS、soft/hard return 改善。相同参数步长不保证相同 KL；这是局部方向有效性，不是调参后优化器的最终排名。
- 所有 forward error 与梯度关系都是固定 Policy 的 checkpoint 观察性比较；5 个模型属于同一训练轨迹，不宣称独立样本或因果关系。

先同 batch32 跑 2 gradient seeds、2 reference seeds、2 directions 的 qualification。基于各阶段实测成本乘 1.3 余量，确认剩余预算和显存后才执行完整配置。新增分配时间上限 1 GPU-hour，包含初始化、qualification、失败及清理；此前 .6055532683 GPU-hour 收敛/复核和更早 .310985655 GPU-hour 保留单列，训练成本仍属于原始训练账本。未运行结果不填零。

输出应包含冻结配置/输入哈希、原始物理 tapes、每次 gradient 投影/范数、完整均值文件哈希、FD 正负 return、Prediction CSV、配对更新 CSV、阶段计时和显存峰值。绘图/结论从原始数据重算。

## 小步长补充对照（在主实验结果揭晓前固定）

资源验收显示，L2=.003 的三种最终模型方向在初始输入上的动作 RMS 约 .0125–.0134；这不是“很小的策略变化”的充分保证。保留原主实验，补充固定 WM4000 的 BPTT/PPO-WM 与 PPO-Sim 三个完整均值方向。该补充不参与主参考的筛选或替换。

使用独立 calibration seed91001 的原生物理 H32 tape，固定其全部 observation/goal，按相同 batch shape 求解各方向的参数步长，使平均 action RMS 为 .0003/.001/.003（主值 .001，容差1%）。固定 σ=.1、29维动作时，对应平均 Gaussian KL 约 .0001305/.00145/.01305。KL 只定义在这条校准输入分布上，不是整条更新后轨迹的 KL。

用主实验同样的16个 held-out noise seeds 做新的物理基线和配对更新评测，不重新估计梯度。参数 L2 上限 .1、扰动量化误差2%，超出则保留失败。先等主实验成功结束，使用实测物理 rollout 时间加30%余量及240秒校准/初始化预留；只有剩余的同一1GPU-hour额度足够才启动。保留所有已用成本，不开启新额度。

## 独立物理 score 参考（补充，非主终点）

主批次完成8方向物理参考后，预设主尺度只有1/8通过筛查。已采样梯度的事后配对分析提示：在WM4000上，相同噪声下的物理score与WM score高度相关。为独立检验这一观察，补充64个新seed101001..101064，与训练梯度、有限差分参考和校准seed完全分离，每个仍是batch32/H32。

固定系数1构造 `g_corrected = g_score_physics - g_score_WM + g_pathwise_WM`。当同一WM随机目标满足微分与期望交换条件时，后两项的期望抵消，因此参考针对物理目标，而不是把WM梯度当真值。保留三项及修正估计的逐draw投影、完整均值和方差；即使方差变大也报告，不拟合系数、不筛选有利方向。CPU使用不同动力学增益的解析Gaussian例子验证期望等于物理梯度。

该参考仍是有限样本Monte Carlo估计，需同时报告SE和原始物理score估计；它不是解析Jacobian、不是新的训练seed，也不替换原有限差分。使用held-out reference时可评估主批次各WM梯度的误差，但不得将其不确定性隐藏。补充与KL评测一起运行，费用包括64次新增物理rollout和配对WM梯度，仍受原1GPU-hour累计上限约束。

还预先固定7个完整参数向量比较：最终BPTT对PPO-Sim、BPTT对PPO-WM、PPO-WM对PPO-Sim，以及四对相邻checkpoint的BPTT。对来自主批次、已固定的两个估计A/B，利用独立物理参考g估计 `||A||²-||B||²-2g·(A-B)`，即两者相对物理期望梯度的平方误差之差，未知的真实梯度范数被消去。负值表示A的条件平方误差较低。逐noise保存该标量、用64个batch估计SE；同时用原始物理score和控制变量参考计算。该检验针对这一次固定的有限采样梯度，不表示BPTT或PPO算法的普遍期望准确性，也不把noise SE当训练seed置信区间。

对上述7个固定比较同时报告以64个独立noise batch为重采样单位的max-|t| bootstrap近似95%区间（10000次，分析seed111001）。所有比较共用重采样索引，保留协方差；仅将该组同时区间用于比较，原始score结果作交叉检查。该近似区间不涵盖有限精度、目标选取、训练seed或状态分布的不确定性。

## 分辨率失败后的执行记录

首次补充已完成64组独立参考并保存完整均值，在第一个最小RMS校准检查处退出；原日志、223.212秒费用及已完成参考均保留。恢复时继续采用原1%RMS容差、2%参数扰动量化门槛，不放宽阈值。逐个尝试全部9个校准条件，对未通过者保存实测RMS、实际参数步长及舍入误差，标为unresolved并不执行对应物理更新；其余条件继续评测。

恢复读取已完成且校验通过的64组参考，重新收集校准与评测基线；参考采样不重复、不重新记为新增样本。恢复预算从同一1GPU-hour中扣除主批次、初次入口失败及这次补充失败，记录完整费用谱系。此改动完善失败记录和继续执行，不修改科学终点或验收门槛。
