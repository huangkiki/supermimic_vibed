# Policy 参数梯度：接触历史分组

本实验扩展到两个各含 8 个起点的批次，使用同一个已训练 Policy/WM/Critic/Adam checkpoint，H=32、FP32、固定噪声种子 9001。每批计算其平均目标的一个完整参数梯度；8 个起点不是 8 个独立梯度样本。每批包含同样的 3 段 run、1 段 sprint、4 段 walk，具体帧不同，因此组间差异不能解释为接触状态的因果效应。

## 状态库与预先指定的选择规则

初始计划见 `experiments/ablation/configs/policy-contact-coverage.json`：在旧 64 状态库中，排除此前分析的 0/1，按库顺序选择同时包含两类接触历史的前 8 段 motion，各取每类第一个起点。选择不使用梯度或 return。接触用实际 MuJoCo-Warp 最后一个物理子步的有效 geometry pair，经 body 映射合并接触片；没有接触力或子步事件真值。

旧库缺少完整 raw replay history。64 状态重放后，最大末端偏差 0.0009658634662628174，超过保持原末端所要求的 1e-5；原计划在梯度计算前停止，失败日志和 19.541911943 秒 GPU 占用保留。没有调宽原门槛。

修订计划 `policy-contact-coverage-replay.json` 在新梯度计算前固定：用相同 reset frames 和固定 16 步 parent action tape 生成**另一个派生状态库**，记录实际 raw history、末端及接触，同时保存 parent hash、原末端和逐状态漂移。它是固定动作重放，不是重新采集的 closed-loop policy prefix，也不是旧库精确末端或历史训练 minibatch。旧库文件保持不变。重新核对原预选索引的 motion 和过去接触标签，任一改变都在梯度计算前拒绝。

`--replay-prefix-bank` 只允许和 `--prepare-history-only` 一起使用；常规参数梯度最多 16 起点的限制保持不变。旧库及派生库结果分开报告。

## 指标与解释

- 3 个归一化更新方向（raw gradient、clipped gradient、historical Adam）与 8 个随机方向；epsilon=0.001/0.003/0.01，3 次相同初始状态的 baseline 重复。
- 原始 `policy_update()` 的 H32 closed-loop reward + soft survival + critic bootstrap 为匹配代理目标；硬终止物理 return 单独报告更新有效性。
- 随机方向相关性、符号一致率按组、目标、尺度分别汇总，保留未解析方向；数值解析筛选可能造成选择偏差。不能把选定梯度方向混入随机方向相关性来抬高结果。
- 过去标签只使用保存的 prefix；起点的对称 ±2 步邻域使用未来观测补全；未来 H32 窗口标签另列。初始状态是最后一个 prefix contact，不是第一个后续状态。
- 保存全部 baseline、正负扰动和实际 Adam 的接触轨迹，以及每个起点的物理改进。一个批次梯度不复制成每个起点的独立梯度预测。

## 可复现入口

准备派生库（在检查 GPU UUID、可用显存、剩余累计预算和磁盘后执行）：

```bash
python -m experiments.ablation.fidelity --probe-space policy \
  --checkpoint "$CHECKPOINT" --bank "$PARENT_BANK" --output "$OUT/history64" \
  --sample-indices $(seq 0 63) --prepare-history-only --replay-prefix-bank \
  --record-contacts --precision fp32 --gpu 2 --gpu-uuid "$GPU_UUID" --max-seconds 75
```

每组使用同一个派生库，按配置中明确的索引运行：

```bash
python -m experiments.ablation.fidelity --probe-space policy \
  --checkpoint "$CHECKPOINT" --bank "$OUT/history64/policy-bank.pt" \
  --output "$OUT/stable_contact" --sample-indices 55 20 21 5 43 8 29 31 \
  --gradient-horizon 32 --probe-seed 9001 --directions 8 \
  --epsilons .001 .003 .01 --repeats 3 --precision fp32 --record-contacts \
  --gpu 2 --gpu-uuid "$GPU_UUID" --max-seconds 165
# switch_nearby: indices 19 2 3 41 7 44 11 13; 独立输出目录。
python -m experiments.ablation.analyze_policy_contacts \
  --root "$OUT" --plan experiments/ablation/configs/policy-contact-coverage-replay.json \
  --output "$ANALYSIS"
```

原生运行结果和成本将在完整张量验收后记录。
