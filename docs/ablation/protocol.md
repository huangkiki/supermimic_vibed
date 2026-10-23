# SuperMimic：World Model 与优化器消融协议

状态：实现与原生运行资格验证进行中。正式多种子结果尚未产生；本文件是预先声明的协议，不是结果。

## 研究基线与论文边界

代码基线：`huangkiki/supermimic_vibed@a1508d42c9d1699a52e8233ab16d08c488441ab9`，独立复现，非论文官方代码。审阅了[论文 v1](https://arxiv.org/pdf/2610.28480)第 3、4 节、附录 A/B/C/D/F；附录 E 用于区分 BONES 与本实验的 LAFAN 数据。

论文 §3 的关键约束为 k=8 自回归 WM、H=32 BPTT、冻结 WM 权重但保留输入 Jacobian、软生存权重及 critic bootstrap。Figure 4(a) 是 40 段 LAFAN 的物理评估；Figure 5(b) 是 walk/run/sprint 的训练回报对 imagined steps，不能直接当作物理评估收益或端到端成本。论文没有给出 Figure 5(b) 的逐文件名单、PPO-WM 全部超参数及可核对的预训练账本。因此这里做受控复现实验，不宣称精确重现其 PPO 数值。

原仓库差异保留并披露：自碰撞关闭、额外 6D 参考朝向观测、局部坐标动力学与 RunningNorm、解析 reward/obs/termination、启动时摩擦/CoM 随机化及逐回合关节偏差、10 秒训练 timeout。默认实验使用 FP32/TF32，原训练默认 bf16；同一对照组精度完全相同。原入口和源文件不改动。

## 已审计的代码与具体修改

| 原文件 | 审计结果 | 新增实验实现 |
|---|---|---|
| `supermimic/train.py` | collect→fit→BPTT；导入即执行；计时在初始采集后；无 imagined 步数或阶段计时 | `experiments/ablation/train.py` 独立入口，复用原模型与 MDP，端到端账本 |
| `supermimic/networks.py` | 确定性 Policy、WM 输入/输出归一化、critic | `ppo.py` 保留原 mean network，仅加 29 个 log-std 参数 |
| `supermimic/env.py` | Newton/MuJoCo-Warp 原生链路；返回 reset 前 `s1`/term/done | PPO timeout 从返回的终态 bootstrap；trace 不跨 reset |
| `supermimic/mdp.py` | 99D 观测，解析 reward/soft/hard termination 和 ξ | 全部方法直接复用；无单独 PPO reward |
| `supermimic/buffer.py` | 窗口 mask 防止跨回合，history 限制在本回合/缓冲区内 | 同冻结实验加载相同 replay，限制采样循环次数 |
| `supermimic/motion.py` | 30→50 Hz、FK、未来参考帧 | 按显式类别选择18段或全部40段，保存每文件 SHA |
| `supermimic/evaluate.py` | 按整段长度归一化；同进程连续评多个 checkpoint | 新 evaluator 每 checkpoint 新进程、reset 后 refresh、完成槽位安全复位；所有方法用相同规则 |

新增 `common.py`（配置/身份/计时）、`run_suite.py`（资源预检和总预算）、`analyze.py`（拒绝混硬件/数据，生成 CSV/曲线）、`fidelity.py`（原生数值参考）、`fidelity_math.py`（稳定性筛选）。

## 两个不同的比较问题

### 系统端到端比较

- A `ppo_sim`：原生 MuJoCo-Warp 物理 rollout，PPO+GAE。没有 WM 训练开销。
- B1 `ppo_wm_online`：持续物理采集、WM 更新、imagined PPO；策略阶段完全 no-grad 穿过 WM。默认使用与物理 PPO 相同的硬终止阈值。
- C `supermimic`：原生 collect→fit WM→BPTT+Critic；保留 H=32、软生存权重和 bootstrap。
- B2 `ppo_wm_frozen_hard`：硬终止与PPO-Sim一致。完整计入独立 pretrain 的采集/拟合成本；后续不再采物理训练数据、不改 WM 或其归一化。

系统比较报告完整实现的效率。BPTT 的软终止、PPO 的随机动作、Dyna replay starts 与 PPO-Sim 连续 rollout 等差异仍存在，不能把系统差值全部归因于某一个因素。

### 相同 WM 的优化器机制比较

另行运行的辅助条件 `bptt_wm_frozen` 与 `ppo_wm_frozen`（soft） 加载**相同 seed 的同一 bootstrap.pt**，共享 WM、其归一化、replay、初始 deterministic mean、critic、H、goal/history、reward 和软生存权重。两者各自更新 policy/critic；policy rollout 没有进入 WM 的参数更新。

该配对的主要剩余差异是策略优化器与随机/确定性动作。它估计冻结模型下的算法收益，不与在线 WM 的成本混算。单独的 `ppo_wm_frozen_hard` 条件用于硬终止敏感性分析，不能混入同目标机制曲线。

PPO 用未裁剪 Normal(mean, exp(log_std))，记录真正采样动作的 log-prob。执行器原有物理限幅继续生效。初始 std=0.1 是为与原 WM 采集噪声对齐的实验选择，**不是已验证最优的 PPO 设置**；必要时对0.1/0.3/1.0、clip/epoch/KL 做等预算开发集敏感性试验。所有主网络宽度不变。PPO actor/critic 与对应 BPTT 使用相同学习率、critic 轮次、gamma/lambda；PPO新增 clipping、KL停止和可配置 entropy。

PPO rollout 和重复 epochs 期间不更新归一化统计；收集到的 obs/action/old log-prob/target 全部 detach。物理 term 禁止 bootstrap；timeout 允许从 reset 前 s1 bootstrap，但 GAE trace 在任何 reset 截断。冻结机制组只冻结 WM；critic 可以更新自己的归一化。

## 数据、种子与评估

源数据版本 `lvhaidong/LAFAN1_Retargeting_Dataset@ce1572906efe6157840e8474d5a0d7aa87481e74`，40 CSV、264705 原始帧；本次从已有缓存重新验证79个源文件，并与该固定版本的实时 Hugging Face 对象核对（31个LFS SHA-256、48个Git blob SHA-1）；检查所有 CSV 的36列格式。完整动作清单随每个运行 manifest 保存。

第一轮为全部12个 walk、4个 run、2个 sprint（18段）；这只是与论文类别对齐，论文未公开逐文件名单。随后 `subset=all40`。正式种子0/1/2，模型、策略、critic 分别使用 seed+111/+222/+333，使增加 WM 或 Gaussian 模块不会改变初始 mean 权重。

每个 checkpoint 在新物理进程中：每段从第一帧开始，无动作/观测噪声、无DR、摩擦1；用 mean policy；完整播放或训练失败条件终止。报告：逐 clip 的 reward sum、按完整参考长度归一化的 return、实际生存步数和 failure。主聚合是 clip 等权均值，另报按帧加权结果。训练 motion 的评测衡量覆盖，不是未见动作泛化。所有硬件/输入/模型资产哈希/精度/核心代码与评估协议不同的记录拒绝放在同一组图中。

不把短截断 rollout、逐存活步平均 reward、imagined return 或工具退出码当作任务成功。初始和最终 checkpoint 至少各评一次；正式曲线按预先配置的 checkpoint 间隔评估。重复性复评另存，不当作独立随机种子。

## 成本与公平比较

每个训练控制步是50 Hz的一步，物理内含4个5 ms substeps。日志计数为 env数×vectorized steps；模型拟合处理的训练窗口不冒充 imagined policy steps。

保存：训练物理步数、imagined policy步数、收集秒数、WM拟合秒数、imagined forward秒数、policy+critic优化秒数、总墙时、GPU-hours、PyTorch峰值显存、实际配置。阶段计时前后 CUDA synchronize。计时包括初始化、数据加载、calibration/warmup和编译；进程级 runner 另记录完整 allocation 时间及失败尝试。

启用compile时，PPO-WM与BPTT的policy→WM→observation想象单步都编译，物理采集的mean/Gaussian actor也编译；避免把eager PPO-WM和compiled BPTT的差值误归因于优化器。Smoke明确关闭compile，Pilot还须验证编译路径。

独立方法成本：`采集 + WM拟合/更新 + 策略/critic + 初始化/编译/IO + 所有此前checkpoint物理评估`。冻结 WM 的预训练全额计入每个方法；实际 campaign 成本只执行一次共享 pretrain，并用 bootstrap SHA 追踪，二者分列。绝不能用 warm WM 的增量训练成本冒充 cold-start PPO-WM 成本。

分析时传入 `--campaign`，最终checkpoint使用完整训练进程、全部前序评估进程及预训练进程的实际占用秒数，包含启动、写盘和退出。中间点使用checkpoint内计时快照，尚不包含该次checkpoint自身写盘和进程退出；CSV的 `cost_basis` 显式区分。runner先完成训练再逐checkpoint复评，因此中间成本曲线是按“若评估穿插训练”重建的累计成本，不是现场在线完成时间。实际campaign总占用另报，不把重建值冒充实测端点。

图包括 Return vs end-to-end wall-clock / total allocated GPU-hours / training simulator steps。物理评估步数单独记录。单GPU时前两条横轴成比例，这是正确现象。冻结WM的 simulator-steps 横坐标恒定，不据此拟合所谓样本效率斜率。

均值与样本标准差(ddof=1)在各seed的共同支持区间以最近已完成checkpoint左保持，绝不外推，不进行移动平均；单seed不绘标准差带。计划报告首次达到0.6/0.82物理回报的成本；没达到就是未达到，不推算时间。

大规模前：先 native smoke，再相同目标规模的短计时/显存探针；按实测吞吐估算全矩阵时间，显存留15%以上余量；runner要求明确总GPU-hour上限及formal forecast。安装/资产资格开销单列为实验准备，不混成某个方法的训练收益。

## Prediction Error 与 Gradient Fidelity

固定 held-out transition bank，来自独立 probe seed=9001、均衡motion取样、固定参考PD前缀及同一 collector policy。状态/动作/ξ和collector SHA完整保存，从不送回训练replay；这是训练motion上的独立轨迹留出，不是未见motion测试。

同一 bank 对各阶段 WM 计算终点 h=1/8/16/32 的分量 MSE：位置m²、旋转rad²、线速度(m/s)²、角速度(rad/s)²、关节角rad²、关节速度(rad/s)²。另给固定尺度(.3m,.4rad,1m/s,3.14rad/s,1rad,10rad/s)的 scaled MSE；该尺度由本协议声明，不用不同 checkpoint 自己的归一化替代，不声称是论文 Fig3(c) 的同一指标。

本版本实现**动作序列方向**，尚非完整 policy 参数方向。默认32步有限时域、gamma=.99、无critic，避免将critic梯度混进动力学梯度。可用 `--gradient-horizon 1/8/16/32` 另行评估短窗口，使稳定接触阶段不被整个32步窗口内的其他切换混淆；不同horizon代表不同目标，必须分开报告。状态预测仍固定评估1/8/16/32步。WM梯度用fp32且关闭TF32；这种诊断精度与训练精度分开记录。方向为每样本整段动作的单位L2随机方向，另测单位L2预测梯度方向。

中心有限差分：`[J(a+eps*v)-J(a-eps*v)]/(2eps)`；eps=0.003/0.01/0.03/0.1。每次从相同reset状态和完整固定前缀重放，要求初始状态max drift≤1e-5；多次零扰动回报差估计数值噪声。只有信号超过max(5×重复噪声,1e-7)且至少相邻两尺度方向导数相差≤25%才记resolved。其余保留为unresolved，不算符号一致或不一致。

分别保存真实硬终止物理回报FD、匹配WM软生存目标的物理FD；报告符号一致率、Pearson（样本/方差不足留空）、正向微小更新的真实return improvement。FD只是尺度依赖的数值参考，不是MuJoCo-Warp解析Ground-truth Jacobian。

接触来自实际MuJoCo-Warp collision/constraint records的geom-pair集合（dist≤0且efc有效），并明确是最后物理substep的记录，不是足高近似或接触力真值。在±2个控制步的接触集合变化附近标记switch_nearby，否则区分stable_contact与flight。同时记录初始局部标签和对应预测/梯度horizon的窗口标签；窗口内任一时刻接近切换都归入switch_nearby，只有全窗口稳定才归stable_contact/flight。保留整段contact signatures供更细分析；某组没有样本就明确留空。

不同checkpoint的Prediction Error与其物理评估return关系只能先报观察性相关：策略成熟度、数据分布和训练时间共同变化。同输入的模型梯度与真实微步改善更直接检验机制；也不能据此宣称低MSE必然导致更高最终policy performance。
