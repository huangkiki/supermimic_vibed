# 可复现运行入口

在原仓库规定的 Isaac Lab `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` 原生环境中，从仓库根目录运行。先按原 README 准备 G1 MJCF/USD 和完整40段数据。实验代码不要求改动原训练入口。GPU、软件构建、资产和 motion SHA 会保存在每次运行的 manifest。

本次A100隔离安装、CUDA构建与依赖约束见 [environment.md](environment.md)。

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

## 代码进度交付

`main` 保持最新已验证开发进度。每个可验收切片提交并更新PR，当前提交的检查通过后合并；中文README同步代码能力、已完成结果和带核验时间的运行快照。长时间训练未结束不阻止已验证代码合入，但不将待运行或单seed观察写成完整实验结论。

正在运行的实验保留固定执行源码、配置、预算和原始账本；开发分支或`main`更新不能直接覆盖活动checkout。新批次按协议重新核对源码及资源门禁，结果明确记录实际执行版本。

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

## 物理评估分数可用时的费用

`run_suite` 先训练完整方法，再依次评估checkpoint。`analyze` 的中间点回溯到checkpoint训练计时，不能解释为当时已取得物理评估分数。补充导出两个时间轴：

```bash
python -m experiments.ablation.score_timing \
  --evaluations logs/ablation/confirmatory-locomotion-seed1-v2/ppo_sim-seed1/ckpt_*/repeat_*/evaluation.json \
  --campaign logs/ablation/confirmatory-locomotion-seed1-v2/campaign.json \
  --output results/ablation/confirmatory-seed1-ppo-score-timing
```

`score-timing.csv` 的 `reconstructed_checkpoint_seconds` 保留原回溯口径；`score_available_allocation_seconds` 和 `score_available_gpu_hours` 计入该方法完整训练、此前及当前组的所有复评。冻结方法全额计一次共享WM预训练；PPO-Sim不承担未使用的预训练。这里累加的是实测子进程分配时长，不是绝对日期或整个campaign的实际等待时间；不计其他方法、排队和进程之间的编排间隙。`timing-scope.json` 保存这些边界。该轴描述当前离线评估流水线，不能单独当作算法学习速度。

输入必须是从首个已执行评估开始的完整前缀，包含各checkpoint的全部复评、成功完成的进程计费和物理GPU身份；缺失早期评估、未完成复评组或乱序计费会被拒绝。允许导出尚未完成整个campaign的方法前缀，但其训练本身必须已完成。旧曲线和原始数据不改写。

## 正式矩阵

`experiments/ablation/configs/matrix.json` 给出种子0/1/2及18/40段范围。formal JSON的4096环境规模已通过短时资源资格；4000迭代仍是候选学习预算，不是已运行结果。必须先在目标4096环境规模做一至数次完整更新的显存/时间测量，保存forecast和来源。runner对formal要求 `--forecast`，且预计总GPU-hours不超过显式预算、预计峰值显存低于可用显存的85%，核对GPU UUID/种子矩阵并保留磁盘空间。每个进程前再次检查资源。

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

`BUDGET` 必须替换为基于实测并获授权的数值。forecast包含 `gpu`、`gpu_uuid`、`subset`、`seeds`、`projected_campaign_gpu_hours`、`projected_peak_memory_mib`、`projected_disk_gib`、`minimum_disk_reserve_gib`，并应保留原始计时日志和配置身份；手填一个通过门槛的数字不能代替资源资格。

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

后续checkpoint用 `--bank .../heldout-bank.pt`，`--samples`、`--prefix-controller`和`--prefix-steps`与原库一致。库从不参与模型训练。`prediction.csv`与`gradient.csv`保存逐样本、checkpoint、horizon/方向/尺度/接触标签与数值参考有效性。`physical-baseline.pt`保留完整物理状态序列和contact signatures。具体数学目标、单位、噪声筛选与证据边界见 [protocol.md](protocol.md)。

若要避免参考PD在探针边界切换为策略，生成库时添加 `--prefix-controller policy --prefix-steps 16`，并使用新输出目录。此模式跨边界保留同一collector的观测历史；复用时同样传这两个参数。查看 `bank-diagnostics.csv` 的首步/后续动作变化及拒绝采样次数，不能仅凭模式名称认定动作连续或策略成熟。`bank.json`保存库SHA256和首动作重放误差；新配对需要两边相同SHA256，相同路径不再足以证明内容相同。

对同一库的稳定接触短窗口，另外运行 `--gradient-horizon 8`（必要时1），输出到新目录；不同梯度horizon单独分析。缩短梯度窗口不会缩短1/8/16/32步状态预测评估。

另用相同 `--bank` 与 `--reward-objective state_only` 去掉两侧的直接动作平滑奖励，检查full-gradient一致性是否被解析动作正则主导。该辅助目标不替代物理任务评分。

```bash
python -m experiments.ablation.plot_fidelity logs/ablation/fidelity-smoke-v1 \
  --output results/ablation/fidelity-smoke-v1 \
  --evaluations logs/ablation/pilot-v1/supermimic-seed0/ckpt_*/evaluation.json
python -m experiments.ablation.validate_fidelity logs/ablation/fidelity-smoke-v1
```

若使用三次复评，将`--evaluations`改为`.../ckpt_*/repeat_*/evaluation.json`，必须提供每个被探测WM checkpoint的全部复评。该入口复用统一物理分析的GPU/数据/协议与重复完整性检查；先按checkpoint求Return均值和样本标准差，再与固定库的1/8/16/32步误差关联。`prediction-performance.csv/json`保存均值、复评次数、评估标准差、训练seed和各时域MSE；单次评估的标准差为空。图上的误差棒是checkpoint内部复评波动，不是多个训练seed的误差棒。

每轮campaign运行期间保持源码和配置不变；runner会在每个进程前核验快照并拒绝继续混用代码版本。修改后必须新建campaign目录。安装/资产资格、被排除的运行和独立gradient probe分别保留成本，不塞进某个方法的训练收益。

若所有训练都已完整结束，只有进程间资源检查导致评估未跑完，可以使用独立恢复器，保持原执行checkout和原总预算不变：

```bash
# 从修复后的工具checkout执行；--execution-root指向原始、不可变的训练checkout。
python -m experiments.ablation.recover_evaluations \
  --campaign /original/checkout/logs/ablation/campaign \
  --execution-root /original/checkout --dry-run
python -m experiments.ablation.recover_evaluations \
  --campaign /original/checkout/logs/ablation/campaign \
  --execution-root /original/checkout
python -m experiments.ablation.validate_run /original/checkout/logs/ablation/campaign
```

恢复器拒绝未完成训练、失败/未结束的子进程、变更的执行源码、未记账的输出或耗尽的预算；不重训、不覆盖已有评估。停止现场先保存为`campaign.before-evaluation-recovery-*.json`，恢复器代码哈希另记。原生GPU探针曾复现：进程退出、显存释放且无计算PID时，利用率仍短暂为7%。新资源检查只在无计算PID且UUID相同时限时等待利用率回落，最多30秒；其他计算进程和设备变化仍立即拒绝，不能以“等待”绕过资源门禁。

若停止时还有**完全没有启动**的后续训练条件，可以显式追加`--include-unstarted-training`，先用`--dry-run`核对清单。该选项从原执行checkout和原campaign参数重建配置，只允许没有训练分配记录、输出目录或日志的条件。已启动但未完成的训练仍拒绝重跑。冻结条件必须使用已完成且已计费的原pretrain，bootstrap内容须与此前冻结条件一致；恢复过程中再次核验输入。训练和后续评估都计入原总上限，不开新账本。恢复前仍需根据剩余训练与评估重估时间、显存、磁盘，保存资源预检；不能把`--dry-run`成功当作训练已经开始。

2026-10-24的旧seed0矩阵在5.581506 GPU-hours后被原资源门禁拦住：所有已启动子进程均成功，剩余冻结软目标PPO的4次评估，以及从未开始的冻结BPTT训练与5次评估。原执行版本30739d2与14 GPU-hours上限保留，外部恢复器单独记录版本；原退出码1和被依赖门禁阻止的seed1队列也保留。该次退出没有保存失败瞬间的GPU采样，不能确定其瞬时原因；当前空闲不证明当时无其他负载。

## 汇总独立的三个正式训练seed

三个独立campaign均完成训练、物理复评和`validate_run`后，用以下命令合并；当前尚未有完整三seed结果，不应提前运行并把局部结果当成正式统计：

```bash
python -m experiments.ablation.analyze_cohort \
  --plan experiments/ablation/configs/confirmatory-locomotion-plan.json \
  --campaigns logs/ablation/confirmatory-locomotion-seed1-v2 \
              logs/ablation/confirmatory-locomotion-seed2-v2 \
              logs/ablation/confirmatory-locomotion-seed3-v2 \
  --output results/ablation/confirmatory-locomotion-v2
```

入口要求预登记的完整seed/method/checkpoint/repeat矩阵、同一物理GPU/源码/输入/评估协议、原生tensor/checkpoint验收记录、完整训练CSV与进程计费，并重新核对逐clip评分。缺seed、混入开发seed0、重复分配、漏掉复评、改变参数或冻结组共享模型不同都会拒绝合并。该汇总不重新运行物理实验，不能替代原生验收。

除Return/Compute曲线外，生成`final-seed-statistics.csv`（各训练seed复评均值的总体均值和样本SD）、`target-observations.csv`（每seed首次达标或最后观察到的删失成本）、`target-summary.csv`（达标分子/分母及未删失时的成本均值/SD）、`paired-target-costs.csv`（同seed的基线/候选GPU成本比）和`cohort.json`（预登记与原账本哈希、去重实际成本）。独立冻结方法曲线各付全额预训练；实际campaign合计只计一次共享预训练。

主阈值Return≥0.8，辅助阈值0.5/0.7均保留失败率≤0.2约束；只选首次实际评估checkpoint，不插值。任一种子未达标，相关未删失平均达标时间和成本比留空，保留达标比例及逐seed最后观察成本。成本比只有双方所有seed均达标时报告样本均值/SD，不能把成功子集当完整矩阵。`time-to-target`在这里指**按账本核算的训练加评估成本**，不是事后复评成绩实际可见的日历时间；原训练先完成再复评的时序限制仍保留。

## 按原预算串行续跑seed 2和3

当前seed1由已记录PID的独立队列等待旧矩阵验收后启动。可从**独立编排checkout**运行`continue_seeds`，待该队列成功后串行完成seed2、seed3，最后执行上面的三seed汇总。运行中的训练checkout和编排checkout均须保持源码/config不变；后续开发使用另一目录。以下变量需指向实际路径，使用前核实对应进程和文件：

```bash
# EXECUTION_ROOT：完成原生all40-smoke-v1资格的固定训练checkout
# PLAN / FORECAST：该checkout中已预登记的计划和seed1测量预算
# RUNTIME：保存已核验seed1队列、PID和脚本的目录
python -m experiments.ablation.continue_seeds \
  --execution-root "$EXECUTION_ROOT" --plan "$PLAN" --forecast "$FORECAST" \
  --predecessor-queue "$RUNTIME/confirmatory-seed1-queue-v3.json" \
  --predecessor-pid-file "$RUNTIME/confirmatory-seed1-queue-v3.pid" \
  --predecessor-script "$RUNTIME/queue_confirmatory_seed1_v3.py" \
  --gpu 0 --wait-seconds 151200 --output "$RUNTIME/confirmatory-seeds23-v1"
```

等待中的状态文件不能单独证明任务存活：入口核验PID、启动时间和完整argv，允许活进程写JSON时的短暂不完整观测；实际失败、进程消失或PID复用均停止后续启动。前一seed需原范围、原上限、完整费用和原生资格全部通过。seed2/3每次启动前都重新核验同一GPU、可用显存、磁盘和剩余总预算；冻结原训练参数，仅改变seed。单seed上限32、总96 GPU-hours不变，等待时间不冒充GPU占用。资源不足不自动删除文件、放宽预留、重启训练或新增预算。

结果保存在各自`confirmatory-locomotion-seed{1,2,3}-v2`目录。外部队列保存原子更新的`queue.json`、每批资源预检和子进程PID/log，最终`cohort-analysis/`只在全部三seed完成后生成。若队列停止，先检查已存在的campaign与实际进程，保留成功项和失败账本，不重新运行已完成seed。

上述示例使用18段`locomotion`计划。续跑器将计划的`subset`显式传给每次`run_suite --subset`，使训练、forecast和`confirmatory-{subset}-seedN-v2`输出目录对应同一动作集合；不依赖子进程默认子集。CPU回归覆盖`all40`与`locomotion`两条分支，并检查seed编号与原预算不变。以后扩展全40段时仍需独立的已验收计划、forecast和资源预检，不能复用18段预算；也不要为使用新版本而改动正在运行的编排checkout。

## 实测预算与配对梯度统计

```bash
python -m experiments.ablation.forecast \
  --cold-probe logs/ablation/resource-probe-v1 \
  --warm-bptt logs/ablation/resource-warm-bptt-v1 \
  --seeds 0 --output forecast-seed0.json
python -m experiments.ablation.compare_fidelity logs/ablation/paired-components \
  --output results/ablation/paired-components
```

配对目录使用 `gradient-components-{full,state_only}-h{1,8,32}`。配对分析只在相同随机方向、相同尺度、两侧都resolved的交集上比较，另报各自覆盖率；CSV保留所有checkpoint、尺度与接触组。方向/尺度复用同一状态，不能作为独立随机种子。

正式seeds1/2/3使用新测量估算，旧的频繁KL提前停止的PPO耗时不再适用。先准备四个完成运行目录（训练CSV、manifest、completion及父目录campaign）；不同探针设备仅用于预算，不混作方法成绩。估算绑定目标GPU UUID，并要求PPO至少实际完成过全套Actor更新：

```bash
python -m experiments.ablation.forecast \
  --ppo-sim-run logs/ablation/ppo-lr2e-5-v1/ppo_sim-seed0 \
  --ppo-wm-run logs/ablation/tuned-wm-resource-v1/ppo_wm_online-seed0 \
  --bptt-run logs/ablation/formal-locomotion-seed0-v1/supermimic-seed0 \
  --pretrain-run logs/ablation/formal-locomotion-seed0-v1/pretrain-seed0 \
  --gpu-uuid GPU-aa15d3ab-4db3-2464-bff6-af4d7e78c4c9 \
  --seeds 1 --iterations 4000 --checkpoint-every 1000 --evaluation-repeats 3 \
  --evaluation-seconds 600 --output forecast-confirmatory-seed1.json
python -m experiments.ablation.run_suite \
  --phase formal --iterations 4000 --checkpoint-every 1000 --ppo-actor-lr .00002 \
  --evaluation-repeats 3 --seeds 1 --gpu 0 --max-gpu-hours 32 \
  --forecast forecast-confirmatory-seed1.json --output logs/ablation/confirmatory-locomotion-seed1-v2
```

上面的UUID是本次已检查的GPU0；其他机器必须重新测量、估算与预检，不直接改名冒用。seed2/3各用匹配seed的新forecast和新输出目录，同一不可变执行版本；三批上限合计96 GPU-hours。第一批在旧矩阵和all40 Smoke验收后开始。实际seed1的PPO-Sim前段与GPU2的40段资源资格有同主机重叠，GPU0本身未共享给其他训练；这不等于整个主机隔离。保留完整实测时间，不减去推测的干扰成本、不事后剔除较慢seed，最终时间收益必须披露这一限制。预登记端点、删失与开发seed排除规则见[PPO资格](ppo-qualification.md#两候选完成与正式参数冻结)。

已训练collector实验使用固定最终checkpoint4000采集bank，四个WM阶段均复用相同动作序列，不能随WM更换collector。下面复现2026-10-24的六探针配置；先检查GPU/磁盘和checkpoint身份，在新输出目录执行。该轮实测0.188922 GPU-hours，资源预检、0.3 GPU-hours硬上限和完整进程费用见交付包的`run-trained-fidelity.py`与plan；以下命令只说明单探针参数，不自行提供预算监督。

```bash
ABLATION_TRAIN=logs/ablation/formal-locomotion-seed0-v1/supermimic-seed0
ABLATION_PROBE=logs/ablation/trained-policy-fidelity-seed0-v1
# 首次只执行full/H1生成状态库；其余五组增加下方--bank参数。
python -m experiments.ablation.fidelity \
  --checkpoints "$ABLATION_TRAIN/ckpt_001000.pt" "$ABLATION_TRAIN/ckpt_002000.pt" \
    "$ABLATION_TRAIN/ckpt_003000.pt" "$ABLATION_TRAIN/ckpt_004000.pt" \
  --collector "$ABLATION_TRAIN/ckpt_004000.pt" \
  --output "$ABLATION_PROBE/gradient-components-full-h1" \
  --prefix-controller policy --prefix-steps 16 --samples 64 \
  --directions 4 --epsilons .003 .01 .03 .1 --probe-seed 9001 \
  --gradient-horizon 1 --reward-objective full
for ablation_horizon in 1 8 32; do
  for ablation_objective in full state_only; do
    if [ "$ablation_horizon:$ablation_objective" = '1:full' ]; then continue; fi
    python -m experiments.ablation.fidelity \
      --checkpoints "$ABLATION_TRAIN/ckpt_001000.pt" "$ABLATION_TRAIN/ckpt_002000.pt" \
        "$ABLATION_TRAIN/ckpt_003000.pt" "$ABLATION_TRAIN/ckpt_004000.pt" \
      --collector "$ABLATION_TRAIN/ckpt_004000.pt" \
      --output "$ABLATION_PROBE/gradient-components-$ablation_objective-h$ablation_horizon" \
      --prefix-controller policy --prefix-steps 16 --samples 64 \
      --directions 4 --epsilons .003 .01 .03 .1 --probe-seed 9001 \
      --gradient-horizon "$ablation_horizon" --reward-objective "$ablation_objective" \
      --bank "$ABLATION_PROBE/gradient-components-full-h1/heldout-bank.pt"
  done
done
python -m experiments.ablation.validate_fidelity "$ABLATION_PROBE"/gradient-components-*/
python -m experiments.ablation.compare_fidelity "$ABLATION_PROBE" \
  --output results/ablation/trained-policy-fidelity-seed0-v1
python -m experiments.ablation.plot_fidelity "$ABLATION_PROBE/gradient-components-full-h32" \
  --output results/ablation/trained-policy-fidelity-seed0-v1/gradient-components-full-h32 \
  --evaluations "$ABLATION_TRAIN"/ckpt_00[1234]000/evaluation.json
```


## 执行独立训练seed的梯度探针

`run_fidelity_cohort`从预登记方案读取三个seed、四个WM阶段、full/state_only、H1/8/32和扰动尺度，顺序运行18个探针。它必须先通过全部正式训练及物理复评的cohort验收，核对真实checkpoint字节；不会在训练仍进行时启动。当前入口已通过CPU控制流程测试，以及真实未完成训练记录的拒绝检查；尚未运行正式三seed GPU探针。

在独立、固定的分析checkout中准备相同原生环境、数据和资产，设置三个已完成训练campaign目录`SEED1/2/3`，再执行：

```bash
PROBE_ROOT=logs/ablation/confirmatory-gradient-v1
probe_args=(
  --training-plan experiments/ablation/configs/confirmatory-locomotion-plan.json
  --probe-plan experiments/ablation/configs/confirmatory-gradient-plan.json
  --training-campaigns "$SEED1/campaign.json" "$SEED2/campaign.json" "$SEED3/campaign.json"
  --gpu 2 --output "$PROBE_ROOT"
)
python -m experiments.ablation.run_fidelity_cohort "${probe_args[@]}" --dry-run
python -m experiments.ablation.run_fidelity_cohort "${probe_args[@]}"
PROBE1="$PROBE_ROOT/seed1"
PROBE2="$PROBE_ROOT/seed2"
PROBE3="$PROBE_ROOT/seed3"
```

`--dry-run`核验完整训练矩阵与checkpoint SHA后打印命令，不查询GPU、不创建输出或启动进程；通过它不表示资源已合格。实际执行前及每个探针前重新检查GPU UUID、型号、空闲、显存和磁盘。预登记保留每seed0.3、总0.9 GPU-hours；时间预估0.236153/seed，显存预估4096 MiB并要求至少8192 MiB空闲，磁盘每seed1 GiB加20 GiB保留。这些资源余量沿用已验收的开发探针方案，不是正式三seed实测峰值。

每seed首个`full-h1`探针生成状态库，随后五个探针使用同一文件；每次启动前重验源码、方案、训练收据、checkpoint及已生成状态库的哈希。不同训练seed各自建库。失败、超时、输入变化或资源不足即停止，保留原目录，不自动重跑或重置账本。中断会终止本次探针自己的进程组，终止清理的实际耗时也计入费用；不隐藏可能的清理超额，不继续启动后续探针。

输出包含根目录`suite.json`、各`seedN/campaign.json`、六组原始CSV和状态库原生验收。GPU费用只计实际探针子进程分配，包含启动、编译和清理；CPU验收与预检时间另包含在总墙时中。所有seed完成并通过严格联合验收后，才可使用下面的汇总命令生成正式图表。正在运行的训练checkout不需要更新。

## 汇总独立训练seed的梯度分析

`confirmatory-gradient-plan.json`预登记正式seeds1/2/3：每seed使用最终SuperMimic collector的64状态库，在该库上比较WM第1000/2000/3000/4000轮、full/state_only、H1/8/32以及四种扰动尺度。探针使用同一GPU2、fp32、关闭TF32和固定物理reset协议；每次启动仍需重新检查资源。开发seed0六探针实测0.188922 GPU-hours，预留25%为0.236153/seed，登记硬上限0.3/seed、合计0.9。探针费用独立记录，不进入策略训练效率排名。当前尚无正式三seed探针结果。

从独立分析checkout执行下列命令；不要为使用新汇总器而更新运行中的训练checkout。`SEED1/2/3`是三个完整训练campaign目录，`PROBE1/2/3`是分别包含六探针和`campaign.json`的已完成目录。当前命令仅经合成单元测试和已有单seed数据兼容检查，尚未在真实三seed数据上运行。

```bash
python -m experiments.ablation.analyze_fidelity_cohort \
  --training-plan experiments/ablation/configs/confirmatory-locomotion-plan.json \
  --probe-plan experiments/ablation/configs/confirmatory-gradient-plan.json \
  --training-campaigns "$SEED1/campaign.json" "$SEED2/campaign.json" "$SEED3/campaign.json" \
  --probe-campaigns "$PROBE1" "$PROBE2" "$PROBE3" \
  --output results/ablation/confirmatory-gradient-cohort
```

入口先复用严格的训练cohort验收，检查完整物理复评、逐clip分数和成本；再验证每个探针的原生bank哈希验收、原始CSV、训练/探针GPU身份、物理reset/精度、源码、WM/collector内容哈希和六次成功分配账本。不同训练seed必须有各自的collector/bank，同一seed六探针必须共享同一bank。分析只读原始验收文件，不覆盖旧收据。

输出四组逐seed与跨seedCSV：方向符号/相关性及resolved计数、真实微步改善与随机方向对照、Prediction Error–物理Return、各seed内部的阶段相关性；另有三组PNG/SVG和`fidelity-cohort.json`来源/费用收据。标准差是独立训练seed之间的样本SD，三个seed等权。任一seed某接触组为空或相关系数无法定义时，跨seed均值/SD留空并报告defined-seed数量，不静默缩小分母。

Prediction Error仅采用预先指定的`full-h32`探针中1/8/16/32步记录；六个重放探针的微小数值差异不算六组独立样本。微步收益在同一epsilon、同一状态上配对梯度方向与全部随机方向，要求所有起点可重放且方向非零；先对每状态的随机方向取均值，再对状态取均值，最后按训练seed汇总。未通过FD尺度/噪声解析的方向仍保留记录；FD方向一致性使用full/state_only共同resolved交集，微步收益另报告重复return噪声。阶段相关性在每seed的四个checkpoint内部计算，不把checkpoint、状态、方向或物理复评当作额外训练seed，不报告因果或显著性结论。

## 全部40段的目标规模资源测量

40段Smoke采用32环境、H4，不能代表正式4096环境、H32的资源需求。`qualify_all40.sh`复用已通过Smoke的固定训练源码3526437，单独运行下面三个有限范围campaign；脚本和forecast位于另一个目录，不修改正在使用的执行checkout。

| Campaign | 训练范围 | 物理评估 | 估算 / 硬上限 GPU-hours |
| --- | --- | --- | --- |
| PPO-Sim与独立WM预训练 | pretrain 500轮；PPO-Sim 32轮 | PPO初始、最终各一次，全40段 | 0.9549 / 1.05 |
| 在线PPO-WM | 500轮仅训练WM，再32轮策略更新 | 初始、最终各一次，全40段 | 0.8854 / 1.00 |
| SuperMimic | 500轮仅训练WM，再32轮策略更新 | 初始、最终各一次，全40段 | 1.0069 / 1.15 |

总估算2.8472 GPU-hours、硬上限3.2，独立于18段正式三seed的96 GPU-hours预算。估算来自18段实测、额外编译余量及每次900秒的40段评估规划额度，**并非已测得的40段耗时或收敛预测**。各阶段预留显存15000/20000/32768 MiB；磁盘合计9 GiB，此外保留24 GiB。GPU2固定UUID，仅用于资源资格；可能与GPU1开发训练重叠，不能把其吞吐当成GPU0正式对照的效率结果。

先准备独立执行目录：checkout精确3526437，并按环境文档提供相同数据/资产。`ABLATION_QUALIFIED_CAMPAIGN`指向该源码已经原生验收的all40 Smoke；脚本核对完整源码/config快照、原生验收和提交身份，每个campaign前检查GPU、磁盘、forecast，再调用现有runner。输出路径必须全新，任一训练/评估失败即停止，不自动重试或扩大预算。

```bash
# Python环境与ISAACSIM_ASSET_ROOT已按环境文档设置。
# RECIPE_ROOT是包含本脚本及三个all40-resource-*.json的独立、固定目录。
export ABLATION_EXECUTION_ROOT=/absolute/path/to/immutable-3526437-checkout
export ABLATION_QUALIFIED_CAMPAIGN=/absolute/path/to/accepted-all40-smoke
export ABLATION_RESOURCE_OUTPUT=/absolute/path/to/new-all40-resource-v1
bash "$RECIPE_ROOT/experiments/ablation/qualify_all40.sh"
```

每个campaign保存`train.csv`、所有checkpoint、物理`per_clip.csv`、完整费用账本、每秒显存采样、验收和单独的图。独立pretrain的完整成本保留在其所属campaign；它没有被在线两方法共享，也没有虚构成PPO-Sim的必要成本。后续全40段forecast需要这些资源测量、冻结条件资格和成熟策略评估耗时，不能用初始策略很快失败的评估耗时外推成功跟踪成本。这一轮不产生多seed或方法优劣结论。

该资格已完成，实际总0.533098 GPU-hours，三策略均40/40失败；目标规模计时、显存、费用和逐clip来源见[实测结果](results.md#全部40段目标规模资源实测)。原规划上限与实际费用分别保留。

## 全部40段冻结条件的资源资格

已完成的冻结三条件可直接用于后续预算：`forecast`同时接受`--frozen-ppo-hard-run`、`--frozen-ppo-soft-run`、`--frozen-bptt-run`，并要求`--pretrain-run`指向它们共同使用的预训练进程。冻结组使用`actual_run_seconds`而非含继承成本的`train_wall_seconds`；完整预训练在campaign合计中只计一次。仅凭CSV/JSON即可[复算全40段规划及核对输入哈希](all40-planning.md)，不需要恢复bootstrap或启动GPU。当前正式18段预算不变。

本次资格已完成：实际0.340834 GPU-hours，4次训练、6次物理评估，最终策略均40/40失败。[实测记录与费用](results.md#全部40段冻结wm资源资格完成)。以下是复现入口，不表示现在需要重复启动；正式18段三seed矩阵运行期间不追加此类GPU测量。

使用上述已验收的固定3526437执行目录，新增一个独立输出campaign。新500轮pretrain由hard/soft PPO及BPTT三条件共享，各32轮策略，六次物理评估。预估2.4141 GPU-hours、硬上限2.5，峰值28701.6 MiB、磁盘5 GiB加24 GiB保留；预计值由40段在线方法实测phase推导，后续以本次直接测量替代。独立预训练成本不能从总账扣除。

先核验前一资源recipe的`exit.txt=0`、三份原生`validation.json`、执行源码未变及当前GPU/磁盘可用；运行器还会在每个子进程前重新检查资源。`FROZEN_FORECAST`指向新增配置的绝对路径，保持在独立recipe目录，不能把新分析源码覆盖进执行目录：

```bash
cd "$ABLATION_EXECUTION_ROOT"
python -m experiments.ablation.run_suite --phase formal --subset all40 \
  --methods pretrain ppo_wm_frozen_hard ppo_wm_frozen bptt_wm_frozen \
  --iterations 32 --checkpoint-every 32 --ppo-actor-lr 0.00002 \
  --evaluation-repeats 1 --seeds 0 --gpu 2 --max-gpu-hours 2.5 \
  --forecast "$FROZEN_FORECAST" --output "$NEW_FROZEN_OUTPUT"
python -m experiments.ablation.validate_run "$NEW_FROZEN_OUTPUT"
python -m experiments.ablation.analyze \
  --evaluations "$NEW_FROZEN_OUTPUT"/*/ckpt_*/evaluation.json \
  --campaign "$NEW_FROZEN_OUTPUT/campaign.json" --output "$NEW_FROZEN_OUTPUT/analysis" \
  --title 'All40 frozen resource qualification only; 32 policy iterations'
```

本次GPU2资源测量与18段GPU0正式三seed实验分开；不重置18段的32/seed、总96预算。完整40段多seed实验仍待新的资源审查和成熟策略评估时长验证。

## PPO 更新尺度检查与真实轨迹

通用`formal-*`模板保留探索矩阵的原始参数；正式seeds1/2/3使用前面的预登记命令，由运行器将PPO Actor学习率覆盖为`2e-5`。复现正式对照时不能省略这一覆盖。Actor 可通过配置中的 `actor_lr` 独立于 `lr` 调整，缺省继承原值，两个学习率均写入训练 CSV。具体失败证据、固定批次 replay 和短程资格配置见 [PPO资格记录](ppo-qualification.md)。

需要保存可视化证据时，用新的评估输出目录运行：

```bash
python -m experiments.ablation.evaluate --checkpoint "$CHECKPOINT" \
  --output logs/ablation/recorded-eval --record
MUJOCO_GL=egl python tools/render.py \
  logs/ablation/recorded-eval/trajectory.npz logs/ablation/recorded-eval/walk.mp4 \
  --clips walk1_subject1 --seconds 6 --title 'Physical evaluation'
```

录制初始状态及真实终止状态，终止后保持最后姿态，避免渲染抽帧时将已reset的姿态误作失败瞬间。记录/渲染成本单独披露，不混入没有录制的主比较曲线。`--record`不改变评估动作、终止或评分。


按条件选择运行、独立 PPO 学习率、三次物理复评和包含全部评估成本的有界长程命令见 [PPO资格](ppo-qualification.md#有界长程验证与重复评估)。非默认正式范围必须有匹配的 `forecast.scope`，缺失评估重复会使验收失败。

冻结资格完成并通过`validate_run`后，可在保存checkpoint的原生环境中追加CPU身份验收：

```bash
python results/ablation/all40-frozen-resource-v1/verify_initial_conditions.py \
  /path/to/all40-frozen-resource-v1 \
  --output /path/to/all40-frozen-resource-v1/frozen-identity.json
python -m experiments.ablation.analyze \
  --evaluations /path/to/all40-frozen-resource-v1/*/ckpt_*/evaluation.json \
  --campaign /path/to/all40-frozen-resource-v1/campaign.json \
  --output /path/to/all40-frozen-resource-v1/analysis \
  --title 'All40 frozen resource qualification only; 32 policy iterations'
```

身份脚本使用CPU加载和流式SHA256，检查本次固定范围（pretrain500、三条件各32轮）；它不重训、不启动物理仿真，也不将同一bootstrap误称为相同采样轨迹。必须使用实际checkpoint，日志或文件名不能替代张量验收。

## 已完成checkpoint的归档与恢复

为保留后续正式seed的原磁盘预算，三份已完成实验的`pretrain-seed0/bootstrap.pt`已复制到独立数据盘，逐文件SHA256校验后移除远端副本，共12,091,735,329字节。三个原campaign中的44个其他checkpoint保留，inode/大小/mtime未变；53,824个其他文件内容哈希未变。活动训练文件不在此次操作范围内。[三个文件的历史相对路径与SHA256](../../results/ablation/checkpoint-storage/retired-bootstraps.csv) · [验收摘要](../../results/ablation/checkpoint-storage/archive-receipt.json)。

完整交付包中的`checkpoint-archive-index.csv`记录实际归档位置和历史远端路径；`Checkpoint归档与恢复.md`提供分批恢复命令。它也包含此前已归档的49个Smoke/Pilot文件，两批共52个文件。原始训练日志的路径保持原样，不能据此推断大文件仍位于原处。

只读CSV绘图无需bootstrap。重新运行`validate_run`、冻结张量身份检查或依赖旧replay的实验前，先按清单恢复相应完整bootstrap；policy checkpoint不能替代其中的replay。恢复前重新检查当前磁盘余量及正式队列预留，使用保留已存在文件的复制方式，再核对SHA256；不批量覆盖原文件、不通过降低资源门槛腾空间。这里提供恢复说明，本次未执行恢复。

## 完整 Policy 参数梯度入口

新增 `fidelity --probe-space policy` 复用原 `policy_update()` actor 前缀，固定模型、raw history 与随机数；默认 Action Gradient 命令保持兼容。15组原生探针已完成，包含H1/8/32四组合消融和FP32敏感性检查。[结果、CSV与复算](policy-gradient-native-v1.md)。见[代码审计、最小命令、资源门禁和剩余验收](policy-gradient-fidelity.md)。正式队列运行期间，新增诊断只使用独立空闲GPU与独立checkout，先检查资源、给定预算并记录并发时段；不得中断正式队列或占用其GPU。


同策略同目标的辅助随机梯度入口、独立噪声划分和等参数步长协议见[匹配估计器](matched-policy-estimators.md)，已完成的H1/H8/H32数据与原始张量复核见[首批结果](matched-policy-results.md)。该入口不更改原生产训练。
