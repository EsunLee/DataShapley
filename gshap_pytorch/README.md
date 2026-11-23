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
  --devices cuda:0 cuda:1 cuda:3 \
  --parallel
```

每完成一次排列自动保存checkpoint；重复同一命令会从下一个未完成排列继续。

## 4. 汇总

```bash
python scripts/aggregate_results.py --results-dir results/ucf_crime
```

生成逐点mean/std、长度704,459的NaN对齐数组，以及训练/总FLOPs和wall-time的`3×50` CSV。

## 双GPU耗时实测

以下基准使用相同的两个独立组（默认seed 13764、14933），分别在一张GPU上串行、两张GPU上同时运行。两种模式都保留现有的`--parallel-perms`组内并行，且写入带时间戳的新目录，不覆盖正式结果。请在与已完成实验相同的Python环境中执行；避开故障GPU 2。

```bash
cd gshap_pytorch
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/benchmark_two_gpus.py \
  --data-dir ../packed_data --split-dir ../results/split \
  --output-dir ../results/two_gpu_benchmark \
  --devices cuda:0 cuda:1
```

每个子进程通过`CUDA_VISIBLE_DEVICES`只看到分配给它的物理GPU，进程内部使用`cuda:0`。每组默认50次排列，完整基准总共运行4组。`benchmark_summary.json`记录串行和双GPU的端到端墙钟时间、加速比、耗时降低百分比，以及同一seed在两种模式下估值的最大绝对差。首次仅检查命令能否工作时可加`--iterations 2`，但该短跑结果不能代表正式50次排列的加速比。该基准测试的是**两个独立组的总耗时**，并不测试“单组50次排列分布到两张GPU”。

### 同一组拆到两张GPU

下面的基准固定master seed 5922，先在GPU 0完成全部50次排列，再按原派生seed顺序将前25次交给GPU 0、后25次交给GPU 3并行运行。两种模式都使用现有stacked实现，比较端到端墙钟时间、逐排列最终AUC及合并后的边际贡献之和。结果写入独立时间戳目录，不会修改正式结果。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/benchmark_split_group.py \
  --data-dir ../packed_data --split-dir ../results/split \
  --output-dir ../results/split_group_benchmark \
  --devices cuda:0 cuda:3
```

两张GPU的子进程各只看到分配的物理卡，内部均使用`cuda:0`。这个实验只测同一组排列的两卡并行耗时；不会将拆分结果写入正式的`group_seed5922`目录。

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
