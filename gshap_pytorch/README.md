# PyTorch mini-batch G-Shapley

该目录实现 [G-Shapley实施计划](../G-Shapley实施计划.md) 中确认的实验：PLOVAD两层`kernel_size=1` decoder、无动量SGD、batch size 128、3组×50次独立随机排列。

## 环境

```bash
cd gshap_pytorch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
```

正式运行前先填写 [远程运行环境确认清单](../远程运行环境确认清单.md)。

## 1. 学习率校准

```bash
python scripts/calibrate_lr.py \
  --data-dir ../packed_data \
  --results-dir results/ucf_crime \
  --device cuda:0
```

输出位于`results/ucf_crime/split/lr_search.csv/json`。正式运行脚本默认读取其中的`selected_learning_rate`。

## 2. 单次真实规模基准

```bash
python scripts/run_group.py \
  --data-dir ../packed_data \
  --results-dir results/benchmark \
  --seed 5922 \
  --device cuda:0 \
  --iterations 1 \
  --learning-rate SELECTED_LR
```

检查`costs_iteration.csv`、效率误差和wall time后，再启动正式任务。基准与正式结果使用不同目录，避免把1次基准误当作正式组的checkpoint。

## 3. 正式运行

单GPU串行：

```bash
python scripts/run_all.py \
  --data-dir ../packed_data \
  --results-dir results/ucf_crime \
  --devices cuda:0
```

三GPU并行：

```bash
python scripts/run_all.py \
  --data-dir ../packed_data \
  --results-dir results/ucf_crime \
  --devices cuda:0 cuda:1 cuda:2 \
  --parallel
```

每完成一次排列自动保存checkpoint；重复同一命令会从下一个未完成排列继续。

## 4. 汇总

```bash
python scripts/aggregate_results.py --results-dir results/ucf_crime
```

生成逐点mean/std、长度704,459的NaN对齐数组，以及训练/总FLOPs和wall-time的`3×50` CSV。

## 合成冒烟示例

```bash
python scripts/make_synthetic.py --output-dir /tmp/gshap_synthetic --n 1000
python scripts/calibrate_lr.py \
  --data-dir /tmp/gshap_synthetic --results-dir /tmp/gshap_results \
  --feature-file X.npy --label-file y.npy \
  --test-size 100 --lr-val-size 100 --lr-train-size 500 \
  --batch-size 32 --device cpu --skip-confirm
python scripts/run_group.py \
  --data-dir /tmp/gshap_synthetic --results-dir /tmp/gshap_results \
  --feature-file X.npy --label-file y.npy \
  --test-size 100 --lr-val-size 100 --batch-size 32 \
  --iterations 2 --seed 5922 --device cpu
```

## 输出与口径

- `group_shap.npy`是组内已完成排列的在线均值。
- `scores_iteration.npy`每行是固定初值0.5和最终AUC。
- `costs_iteration.csv`同时包含单次及累计FLOPs、训练/评估/总wall time。
- batch贡献为`AUC增量 / 实际batch大小`，每次排列均检查`sum(marginal) ≈ final_auc - 0.5`。
- 结果是mini-batch均摊近似，不是`batch_size=1`的严格逐点G-Shapley。
