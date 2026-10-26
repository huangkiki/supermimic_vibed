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
