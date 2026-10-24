# 正式 seed 1 PPO-Sim：评估分数可用时的费用

补充原[完整 PPO-Sim 结果](../confirmatory-seed1-ppo-v1/)，不修改其曲线或输入。五个 checkpoint、每点三次物理复评，仍只有一个独立训练 seed。

流水线先完成4000轮训练，再评估0/1000/2000/3000/4000轮checkpoint。因此第0轮的回溯费用是137.005955秒，但取得这组三次分数时，该方法已累计使用7790.426143秒进程分配。两个轴在最终点一致：12534.245213秒，即3.481734781 GPU-hours。

- `score-timing.csv`：同时保存回溯费用、分数可用时的累计进程费用、物理Return及复评SD。
- `timing-scope.json`：明确费用边界。累计进程费用不是绝对日期，不含其他方法、排队等待及进程之间的编排间隙；这个轴描述离线评估流水线，不能单独用于算法学习效率排名。
- `verification.json`：原包37文件SHA核对、15复评/16进程核算、从压缩包重新生成CSV字节一致；原发表数据保持不变。

在仓库根目录、已安装分析依赖的环境执行：

```bash
mkdir -p work/score-timing-reproduction
tar -xzf results/ablation/confirmatory-seed1-ppo-v1/raw-data.tar.gz \
  -C work/score-timing-reproduction
python -m experiments.ablation.score_timing \
  --evaluations work/score-timing-reproduction/confirmatory-seed1-ppo-v1/ppo_sim-seed1/ckpt_*/repeat_*/evaluation.json \
  --campaign work/score-timing-reproduction/confirmatory-seed1-ppo-v1/campaign.json \
  --output work/score-timing-reproduction/recomputed
cmp results/ablation/confirmatory-seed1-ppo-score-timing/score-timing.csv \
  work/score-timing-reproduction/recomputed/score-timing.csv
```

不需要GPU或模型权重。本次验收实际执行了解包、逐文件SHA验证和复算，并非只做静态检查。PPO-Sim没有使用共享WM预训练，不将该费用加入方法成本。
