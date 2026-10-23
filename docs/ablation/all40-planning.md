# 全40段长程实验的预算规划

**尚未启动全40段正式多seed实验，也未采用新的GPU预算。** 下表只将已完成的目标规模资源测量外推到六条件各4000轮，不能预测收敛、Return或达到同等策略质量的时间。当前18段正式实验仍使用原执行版本和32 GPU-hours/seed、总96的上限。

## 使用实际冻结组耗时

原估算从在线WM的rollout和策略优化阶段推断冻结组时间。现在hard/soft PPO和BPTT冻结组均已完成32轮目标规模资源测量，允许直接使用各自完整训练进程分配，包括加载、首次编译、存盘与退出；只对尚未执行的轮次外推。

冻结日志的`train_wall_seconds`含继承的预训练成本，`actual_run_seconds`才是本进程内的时钟。直接估算采用：

```text
冻结训练秒数 = 已测完整进程秒数
             + (目标轮数 − 已测轮数) × 最后30个暖态间隔的最大秒数
campaign训练秒数 = 一次完整共享预训练 + 三主方法训练 + 三冻结方法训练
```

冻结条件必须一起提供；其bootstrap SHA须一致，路径须指向所选预训练进程的输出，全部CSV行与completion中的物理采集、WM拟合及Simulator Steps须保持在该预训练账本上。继承的墙时也须与预训练completion一致。该检查核对日志身份与费用，实际checkpoint张量身份由[单独的原生验收](../../results/ablation/all40-frozen-resource-v1/frozen-identity.json)证明。

四个原始输入目录的估算入口仍保留，未提供冻结组三目录时明确标记为在线阶段推断。直接模式用`frozen_cost_basis=direct_process_allocation_plus_warm_intervals`，来源、使用的时钟及每组计时均保留在JSON中。

## 同口径规划对比

两列都使用冻结资源批次自己的500轮预训练：完整进程290.172801秒，只计一次。硬件均为A100-PCIE-40GB、4096环境、H32、k8、PPO Actor lr=2e-5；目标GPU为测量的GPU2 UUID。每seed六条件、5个checkpoint、每点3次新进程评估，共90次。**每次物理评估1500秒仍是假设额度，尚非成熟40段策略的实测值。**

| 单seed规划项 | 在线阶段推断冻结组 | 冻结组直接实测外推 |
|---|---:|---:|
| hard PPO冻结训练，秒 | 4159.787 | 3803.116 |
| soft PPO冻结训练，秒 | 4159.787 | 3758.870 |
| BPTT冻结训练，秒 | 4423.563 | 9283.669 |
| 全部训练及一次共享预训练，GPU-hours | 8.700408 | 9.839997 |
| 物理评估额度，GPU-hours | 37.500000 | 37.500000 |
| 含25%时间保留的合计，GPU-hours | 57.750510 | 59.174996 |

直接估算较高主要来自冻结BPTT第15轮的2.231403秒间隔：其中imagined rollout为1.777839秒，策略优化为0.451386秒。最后30轮平均完整间隔为0.834994秒，但本预算规则一直使用最大值，保留这条较慢观测，不删除或换成均值。其原因尚未归因；不能把预算差额称为测得的算法减速。

显存规划为28701.6 MiB（测得的最大峰值加20%），磁盘每seed12 GiB并另留12 GiB。这些工程余量均不是置信区间。旧57.789151 GPU-hours规划还使用了主方法资源批次的另一份401.456787秒预训练，不能将它与新数值的差额全部归因于冻结组估算。历史文件保留；上表用同一份预训练隔离这一记账差异。

[直接估算JSON](../../results/ablation/all40-planning-v2/direct.json) · [同预训练的旧推断JSON](../../results/ablation/all40-planning-v2/inferred.json) · [对照CSV](../../results/ablation/all40-planning-v2/comparison.csv) · [原始输入SHA256](../../results/ablation/all40-planning-v2/input-sha256.json)。这些是规划产物，不能混入Return–Compute实测曲线。

## 仅用CSV和JSON复算

将交付包中的`all40-resource-v1`及`all40-frozen-resource-v1`放在同一原始数据根目录，在仓库根执行，无需GPU、checkpoint或恢复已归档bootstrap：

```bash
PYTHONPATH=. python results/ablation/all40-planning-v2/reproduce.py \
  --raw-root /path/to/raw \
  --gpu-uuid GPU-cc40495e-8e3b-d4c0-894a-63ce2fb079a9 \
  --output /path/to/new-planning-output
```

通用CLI可在原四个测量目录参数外同时增加`--frozen-ppo-hard-run`、`--frozen-ppo-soft-run`和`--frozen-bptt-run`；`--pretrain-run`须改为这组三条件实际引用的预训练目录。只提供其中一部分会拒绝估算。

后续仍需验证成熟策略的全40段物理评估时长，再确定正式seed范围、预算与目标设备。每次长训练前重新检查GPU、显存和磁盘；本次CPU复算没有启动训练，没有修改活动实验的配置或预算。
