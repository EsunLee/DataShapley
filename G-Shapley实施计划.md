# G-Shapley 数据价值评估实施计划（PyTorch 版）

> 状态：**数据已到手并核对完成，实验口径已确认，待实施**（2026-08-14）  
> 任务：使用 Gradient Shapley 评估 UCF-Crime 视频片段的数据价值。正式实验为 **3 个独立组 × 每组 50 次 G-Shapley 随机排列**，共 150 次排列；不是让同一个模型连续训练 50 epoch。

---

## 1. 任务背景与已确认配置

老师布置的任务：使用 Ghorbani & Zou（2019）论文 *What is your data worth?* 的 Gradient Shapley（Algorithm 2），估计视频异常检测训练数据的价值，并统计计算量、运行时间及多组估值的均值和标准差。

- 原始数据：UCF-Crime 视频段特征 `X:(704459, 512)`，二值标签 `y:(704459, 1)`。
- 特征：CLIP ViT-B/16，float32，约 1.44 GB。
- 分类器：PLOVAD decoder：`Conv1d(512→128, k=1) → GELU → Conv1d(128→1, k=1)`；对单片段等价于两层 MLP。已核验 PLOVAD 官方 `master/src/model.py` 第109～113行，两层均明确使用 `kernel_size=1, padding=0`。
- 优化器：无动量 SGD，`momentum=0`、`weight_decay=0`、无 scheduler；学习率在正式实验前按 §2.2 校准。PLOVAD 的 Adam(`lr=1e-3`)只作为原模型配置记录，不用于正式 G-Shapley。
- batch size：128。
- 评价指标：AUC。
- 正式实验：3 个 group seed；每组 50 次 G-Shapley iteration/permutation；总计 150 次独立排列和模型初始化。
- 原仓库 [DShap.py](DShap.py) / [Shapley.py](Shapley.py) 为 TensorFlow 1.12 实现，使用 PyTorch 2.5.1+cu121 / NumPy 1.26.4 / scikit-learn 1.3.0 重写。

### 1.1 数据检查结果

| 项 | 结果 | 结论 |
|---|---|---|
| 文件 | `ucf_train_feat_704459x512.npy` / `ucf_train_label_704459x1.npy` / `ucf_train_meta.json` | 与数据契约一致 |
| X shape / dtype | `(704459, 512)` / float32 | 正常 |
| X 值域 | `[-0.419, 0.843]`，均值 0.0016，无 NaN/Inf | 正常 |
| X 列质量 | 每列 std ∈ `[0.0065, 0.0898]`，无全零列 | 正常 |
| X 行范数 | mean=1.000（0.999～1.001） | 已 L2 归一化，`feature_norm='none'` |
| y shape / dtype | `(704459, 1)` / float32，值为 `{0,1}` | 可用于 BCE loss |
| 标签分布 | 正常 592,665（84.1%）；异常 111,794（15.9%） | 类别不平衡，使用 AUC |
| 独立测试集 | 无 | 固定分层抽取5,000点正式测试集和5,000点学习率验证集 |

### 1.2 交付物

1. `3×50` FLOPs 表和运行时间表：行是3个独立组，列是完成第1～50次 G-Shapley permutation 后的累计值。
2. 3组 G-Shapley 估值的逐训练点 mean/std。
3. 为保持与原始704,459行对齐，额外输出长度为704,459的 mean/std；正式测试集和学习率验证集的位置填 `NaN`。

> 不做数据删除曲线、错标检测等论文应用实验，除非后续另行要求。

---

## 2. 实验口径

### 2.1 一次 G-Shapley iteration 的定义

论文 Algorithm 2 中，一次 iteration 对应一次随机排列。每次 iteration 必须：

1. 使用该 iteration 的派生 seed 重新随机初始化 decoder；
2. 使用已校准学习率重新创建无动量 SGD optimizer；
3. 生成训练子集的一个全新随机排列；
4. 按排列顺序切成 batch size 128，执行一次 one-pass epoch；
5. 每个 batch 更新后，在固定测试集上计算 AUC；
6. 记录当前 batch 引起的 AUC 边际变化；
7. 完成该排列后保存一条边际贡献向量；
8. 每组对50条边际贡献向量取均值，得到该组的 G-Shapley 估计。

因此，本计划中：

```text
1 group = 1 master seed + 50 independent permutations
3 groups = 3 master seeds + 150 independent permutations
1 permutation = 1 newly initialized model + 1 one-pass epoch
```

文档和代码中统一使用 `iteration` 或 `permutation`，不再把“同一模型连续训练50轮”称为本实验的50 epoch。

论文参考代码 `DShap.py` 使用无 momentum 的 SGD，并自动搜索 one-step learning rate。本实验确定使用同类SGD，以减少额外optimizer状态带来的路径依赖，并保持与 G-Shapley 参考实现一致。由于本实验batch size为128而不是1，仍需针对当前batch设定重新校准学习率，不能直接沿用PLOVAD Adam的`1e-3`。

### 2.2 SGD学习率校准

> ✅ **已完成（2026-08-14，由同事完成校准）：选定 lr = 0.1（候选 1e-1），正式实验全部 150 次排列冻结使用。** 以下为校准方法记录（代码保留在 `scripts/calibrate_lr.py` 供复现/存档；正式运行跳过校准步骤）。

参考实现 `_one_step_lr()` 搜索以下8个候选：

```text
1e-1, 10^-1.5, 1e-2, 10^-2.5, 1e-3, 10^-3.5, 1e-4, 10^-4.5
```

原代码每个候选重复10次，并最大化 `mean(test score) - std(test score)`。为缩短校准时间，同时不使用正式测试集调参，本实验采用：

1. 从正式训练候选数据中固定分层抽取100,000点作为 `lr_train_subset`；
2. 另固定分层保留5,000点作为 `lr_val_subset`，与正式5,000点测试集完全不相交；
3. 每个候选使用3个固定校准seed，各进行一次模型初始化、随机排列和one-pass训练；
4. 每次只在one-pass结束时计算一次 `lr_val_subset` AUC，不做逐batch AUC；
5. 排除出现NaN/Inf、loss明显发散或任一次结果无效的候选；
6. 计算 `score = mean(final_auc) - std(final_auc, ddof=0)`，选择score最大者；若差值小于`1e-4`，选择较小学习率；
7. 用选中的学习率在完整正式训练子集上做一次one-pass确认，只检查loss有限且最终AUC高于0.5；
8. 保存 `lr_search.csv/json` 后冻结学习率，全部3组×50次排列使用同一值。

`lr_train_subset`只是正式训练子集的一个视图，校准后仍参与正式训练和估值；`lr_val_subset`不参与正式训练和估值。因此最终约有`704459-5000-5000=694459`个有效估值点。该校准总共24次100K one-pass加1次全训练one-pass，且不逐batch评估，成本远低于正式实验。

### 2.3 batch=128 的归因定义

论文逐点版本使用 `batch_size=1`。本实验已确认使用 `batch_size=128`，因此结果属于：

> **mini-batch Gradient Shapley 的逐点均摊近似**，不是严格 batch_size=1 的逐点 G-Shapley。

设一次更新使用 batch `B`，更新前后测试 AUC 的变化为：

```text
delta = auc_after - auc_before
```

将贡献平均分配给 batch 内样本：

```text
contribution[i] += delta / len(B),  i in B
```

最后一个不足128点的 batch 必须除以其实际大小。不能把完整 `delta` 分别赋给每个样本，否则总贡献会被重复计算。

同一 batch 中的点在单次排列内贡献相同；50次排列会重新打乱，使各点与不同样本组成 batch，从而形成不同的平均估值。报告中必须注明该近似口径。

### 2.4 三组与种子设计

**候选池（共享，已生成并落盘）**：`seeds_100.json`（项目根目录，2026-08-14 生成，100 个全唯一，值域 [5922, 982846]）。生成方式与同事完全一致，比对哈希即可确认双方池子相同：

```python
def generate_100_master_seeds(master_seed=41):
    rng = np.random.RandomState(master_seed)
    return sorted(rng.randint(1000, 999999, size=100).tolist())
```

**分配（一人 3 个，不重叠）**：

| 方 | 组号 | 索引 | 种子值 |
|---|---|---|---|
| EsunLee（本仓库） | 组 1~3 | `seeds[0:3]` | 5922, 13764, 14933 |
| 同事 | 组 4~6 | `seeds[3:6]` | 37446, 54491, 55508 |

- 各自把 3 个落盘 `seeds_used.json`（本仓库已生成），跑完随结果提交；如同事已选定其他 3 个，以实际为准，保证不重叠即可。
- 每个 master seed 确定性派生 50 个 iteration seeds（`np.random.SeedSequence(master_seed).generate_state(50)`）。
- iteration seed 同时控制该次模型初始化和随机排列（torch / numpy 两条独立 RNG 流，互不干扰）。
- 3 组共需 150 个排列；”100 个候选种子”不表示正式实验只有 100 个排列。
- 三组共享同一训练/测试拆分，以保证组间 mean/std 可比较。

### 2.5 测试集拆分及输出维度

`packed_data/` 没有独立测试集。为在保持固定效用函数的同时尽量缩短逐 batch 评估时间，从原始704,459点中按标签固定分层抽取5,000点作为正式测试集，并另抽5,000点作为学习率验证集：

- 测试集大小固定为5,000，split seed固定为0；
- 学习率验证集大小固定为5,000，与正式测试集和正式训练集均不相交；
- 三组共用相同拆分；
- 分层后预计约4,207个正常点、793个异常点，实际数量由程序记录；
- 两个保留集均不进入正式排列、不参与正式梯度更新，也没有训练数据 Shapley 值；正式测试集只用于 G-Shapley AUC，学习率验证集只用于选学习率；
- 剩余约694,459点组成 `train_indices`，只对这些点估值；
- 训练点及两个保留集的精确数量由拆分程序记录。

因此不能在拆出测试集后仍声称获得704,459个有效训练点价值。为方便与原始数据索引对应，输出两种数组：

1. 紧凑数组：长度为 `n_train`，只包含被估值训练点；
2. 全长数组：长度为704,459，训练位置写入估值，两个保留集位置填 `NaN`。

由于没有 `video_id`，当前只能按片段随机分层，无法避免同一视频的相邻片段落入训练集和测试集。该数据泄漏风险可能影响 AUC 及价值排序，必须写入最终报告的局限性，不能声称其对排序没有影响。

### 2.6 初始价值与边际贡献

按照论文参考实现，AUC 的空集合价值固定为：

```text
V(empty) = 0.5
```

不使用随机初始化模型的实测 AUC 代替0.5，以免把初始化噪声集中归给第一个 batch。随机初始化 AUC 可另存为诊断数据。

对于每次 permutation `t`：

```text
s_prev = 0.5
for batch in permutation:
    train_step(batch)
    s = evaluate_auc(test_set)
    delta = s - s_prev
    marginal_t[batch] += delta / len(batch)
    s_prev = s
```

应自动验证效率/望远镜恒等式：

```text
sum(marginal_t) ≈ final_auc_t - 0.5
```

误差仅应来自浮点精度。如果使用间隔评估等额外近似，则必须单独说明误差来源。

### 2.7 核心伪代码

```text
split = stratified_three_way_split(
    X, y, test_size=5000, lr_val_size=5000, seed=0
)

run_group(master_seed):
    group_sum = zeros(n_train)

    for iteration in 1..50:
        iteration_seed = derive_seed(master_seed, iteration)
        set_seed(iteration_seed)

        model = newly_initialized_plovad_decoder()
        optimizer = SGD(model.parameters(), lr=selected_lr,
                        momentum=0, weight_decay=0)
        permutation = random_permutation(n_train, iteration_seed)
        marginal = zeros(n_train)
        s_prev = 0.5

        for batch in permutation.split(batch_size=128):
            train_step(X_train[batch], y_train[batch])
            s = auc(model, X_test, y_test)
            marginal[batch] += (s - s_prev) / len(batch)
            s_prev = s

        assert sum(marginal) ≈ s_prev - 0.5
        group_sum += marginal
        group_shap = group_sum / iteration
        record_costs_and_time(iteration)
        save_iteration_checkpoint()

    save(group_shap)

aggregate_groups():
    values = stack([group_1, group_2, group_3])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
```

### 2.8 断点续跑

- 每完成一次 permutation 原子保存一次 checkpoint。
- checkpoint 包含：已完成 iteration、组内累计边际贡献、成本表、seed信息及完成状态。
- 由于每个 permutation 都重新初始化，恢复时从下一个 permutation 开始，不需要保存上一模型和 optimizer。
- 已完成的 permutation 不重复运行；同一配置和 seed 的中断续跑结果应与无中断运行一致。

---

## 3. FLOPs 与时间测量

### 3.1 理论计算口径

按乘法和加法各计1 FLOP，decoder 的主要矩阵运算为：

| 项 | 公式/说明 |
|---|---|
| 单样本核心前向 FLOPs | `2×(512×128) + 2×(128×1) = 131,328`；两项分别是两层权重乘加，不含 bias/GELU |
| 单样本核心训练 FLOPs（近似） | `3×131,328 = 393,984` |
| 一次 permutation 的训练 FLOPs | `n_train × 393,984` |
| 一组50次训练 FLOPs | `50 × n_train × 393,984` |
| 三组总训练 FLOPs | `150 × n_train × 393,984` |
| 一次完整测试前向 FLOPs | `n_test × 131,328` |
| 一次 permutation 的测试次数 | `ceil(n_train / 128)` |

上述训练 FLOPs 是常用的 `forward + backward ≈ 3×forward` 近似，只覆盖主要网络计算，不精确包含 bias、GELU、BCE、SGD更新、AUC排序、数据搬运和I/O。

保留5,000点正式测试集和5,000点学习率验证集时，`n_train≈694,459`、每次排列约5,426个batch：

```text
train_flops / permutation ≈ 694,459 × 393,984 ≈ 0.274 TFLOPs
eval_flops  / permutation ≈ 5,426 × 5,000 × 131,328 ≈ 3.56 TFLOPs
eval / train ≈ 13.0
```

因此即使缩小测试集，评估仍约占核心网络 FLOPs 的92.9%。若使用原10%（约70K）测试集，评估将约为训练的184倍；固定5,000点可将该部分缩短约14倍。

### 3.2 统计层次

每完成一次 permutation，记录一行：

```text
iteration,
train_flops_iteration, eval_flops_iteration, total_flops_iteration,
train_flops_cumulative, eval_flops_cumulative, total_flops_cumulative,
gpu_train_time, gpu_eval_time,
wall_train_time, wall_eval_time, wall_total_time,
initial_auc, final_auc
```

- 理论 FLOPs：由固定公式产生。
- THOP：用于报告 MACs/参数量，注意 `1 MAC ≈ 2 FLOPs`，不强制与手算 FLOPs比例落在 `[0.9,1.1]`。
- GPU时间：用 CUDA Event 测量，并在边界正确 synchronize。
- 端到端时间：用 `time.perf_counter()`，包含数据读取、CPU/GPU传输、AUC计算和必要I/O。
- 老师要求的训练时间以端到端 wall time 为主，GPU kernel时间作为补充。

精确 AUC 默认在GPU上计算，测试特征、标签和训练特征在显存允许时常驻GPU。禁止在每个batch后执行 `.cpu().numpy()` 再调用 `sklearn.metrics.roc_auc_score`，以避免每次排列约5,426次GPU同步、传输和CPU排序。每次排列结束时才把贡献和统计结果传回CPU并落盘。若全量训练特征无法常驻显存，则使用 pinned memory、non-blocking transfer和预取；测试集必须常驻GPU。

### 3.3 正式 `3×50` 表的含义

- 行：3个 group seeds。
- 列：`permutation_1 ... permutation_50`。
- 单元格：该组完成前 `j` 次 G-Shapley permutation 后的累计 FLOPs或累计时间。
- 同时保留逐次非累计明细，以回答第1次和第2～50次各自的成本。

### 3.4 评估成本控制

主实验默认每个 batch 后都在固定5,000点测试集上计算精确AUC，以保持边际贡献定义一致。正式运行前只做一次 `1组×1次排列` 基准，分别统计训练更新、测试前向、GPU AUC、I/O和总wall time，然后直接外推150次成本。

若该配置仍不可接受，才能选择以下近似并在报告中明确标注：

- 固定测试子集；
- 每 `k` 个 batch 评价一次，并将区间增量均摊；
- 使用更高效的增量指标。

这些设置会改变估计结果，不能在不同组间使用不同配置。默认不做 `clip_deltas` 和 epoch L1 归一化，因为它们会改变估计量并破坏效率恒等式。

---

## 4. 运行方式

- 多卡：每组绑定一张独立GPU，可并行运行3组。
- 单卡：默认串行运行3组，保证时间可比较，并避免多进程争抢GPU和数据带宽。
- CPU：作为调试兜底，不建议用于正式全量实验。
- 每组使用独立输出目录和 checkpoint。
- 若要在单卡并发，必须先比较1/2/3进程吞吐量；正式计时不得混用串行和共享GPU并行结果。

### 4.1 组内 50 排列并行（`--parallel-perms`，正式实验推荐）

串行跑一组 50 排列 ≈ 8.5 分钟，瓶颈是每次 kernel 只算 1 个 128 点 batch（GPU 利用率 0.12%）。`--parallel-perms` 把一组内的 50 个排列**同时**执行：

- 机制：50 份 decoder 参数堆叠成 `(50, …)` 张量（`w1 (50,512,128)`、`b1 (50,1,128)`、`w2 (50,128,1)`、`b2 (50,1,1)`），每 batch 步把 50 个排列各自的 128 点收集成 `(50,128,512)`，前向/反向全用 `bmm`；SGD 更新逐流 `add_(grad, alpha=-lr)`（与 `torch.optim.SGD` 同一运算）；评估把 5,000 点测试集广播成 `(50,5000,512)` 一次出 50 组 logits，AUC 用 `binary_auc_batch` 批量计算。
- 语义不变：每流 batch=128、每 batch 后评估、`ΔAUC/len(B)` 逐点分摊、效率恒等式逐流校验；种子与排列生成与串行完全一致。
- 一致性：`binary_auc_batch` 与逐流 `binary_auc` 逐位一致（同一套 float64 整数精确秩运算）；模型前向/反向若 cuBLAS 与 Conv1d kernel 收敛路径不同，可能在最后 1 ulp 级别有差异——由 `tests/test_stacked.py`（atol=1e-12）与 `scripts/verify_stacked.py`（打印 bit-identical 判定）验证。
- 时间语义：`costs_iteration.csv` 每行的 wall/gpu 时间是**组墙钟按排列数均摊**（累计列 = 组墙钟）；`meta.json` 记录 `parallel_perms: true`、`wall_group_seconds`、`time_semantics`。
- 效果：一组 ≈ 15-30s（实测为准，评估 FLOPs 是训练的 13 倍、主导耗时）；3 组绑 GPU 0/1/3 并行总墙钟 ≈ 1 分钟以内。
- 使用：`run_group.py --parallel-perms`；断点续跑时剩余排列继续堆叠执行，`group_sum` 累加顺序不变。

---

## 5. 建议目录结构

```text
gshap_pytorch/
├── README.md
├── requirements.txt
├── gshap/
│   ├── __init__.py
│   ├── config.py
│   ├── seeds.py
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   ├── flops.py
│   ├── trainer.py
│   └── io_utils.py
├── scripts/
│   ├── run_group.py
│   ├── run_all.py
│   ├── make_synthetic.py
│   ├── calibrate_flops.py
│   └── aggregate_results.py
├── tests/
│   ├── test_model.py
│   ├── test_flops.py
│   ├── test_attribution.py
│   ├── test_data.py
│   └── test_io.py
└── results/
```

数据加载层支持两种模式：

- `split_mode='holdout'`：当前方案，从704,459点固定分层拆分；
- `split_mode='external'`：未来如果取得独立测试特征，可直接替换而不改训练和归因逻辑。

---

## 6. 输出文件

### 6.1 固定拆分文件

`results/ucf_crime/split/`：

| 文件 | 内容 |
|---|---|
| `train_indices.npy` | 原始704,459行中参与训练和估值的位置 |
| `test_indices.npy` | 原始704,459行中作为测试集的位置 |
| `lr_val_indices.npy` | 与正式测试集不相交的5,000个学习率验证点 |
| `split_meta.json` | split seed、数量、类别数量及索引哈希 |
| `lr_search.csv/json` | 候选学习率、3次AUC、mean/std、选择分数及最终学习率 |

### 6.2 单组文件

`results/ucf_crime/group_seed{S}/`：

| 文件 | 内容 |
|---|---|
| `meta.json` | master seed、配置、拆分哈希、状态 |
| `iteration_seeds.npy` `(50,)` | 50个派生 iteration seeds |
| `permutations/perm_XXX.npy` | 每次随机排列；可配置是否全部保留 |
| `marginals/marginal_XXX.npy` | 每次排列的训练点贡献；可在汇总及校验后归档 |
| `group_shap.npy` `(n_train,)` | 50次排列贡献的均值 |
| `scores_iteration.npy` `(50,2)` | 每次的 initial/final AUC |
| `costs_iteration.csv` | 50次排列的逐次和累计 FLOPs/时间 |
| `checkpoint.pt` | iteration边界断点 |

### 6.3 三组汇总文件

`results/ucf_crime/`：

- `shap_mean_train.npy` `(n_train,)`：3组逐训练点均值；
- `shap_std_train.npy` `(n_train,)`：3组逐训练点样本标准差，`ddof=1`；
- `shap_mean_all.npy` `(704459,)`：训练位置为mean，两个保留集位置为 `NaN`；
- `shap_std_all.npy` `(704459,)`：训练位置为std，两个保留集位置为 `NaN`；
- `shap_summary.csv`：有效训练点的mean/std描述统计；
- `flops_3x50_train.csv`、`flops_3x50_total.csv`；
- `time_3x50_train.csv`、`time_3x50_total.csv`；
- `cost_detail_seed{S}.csv`：每组50行明细；
- `seeds_100.json`、`seeds_used.json`。

对训练点 `i`，三组统计为：

```text
mean_i = (phi_i^(1) + phi_i^(2) + phi_i^(3)) / 3
std_i  = sqrt(sum_k (phi_i^(k) - mean_i)^2 / (3 - 1))
```

其中每个 `phi^(k)` 已经是该组50次随机排列贡献的平均值。

---

## 7. 验证方案

| 阶段 | 数据与规模 | 目标 |
|---|---|---|
| 0 冒烟 | 合成 n=5,000，1组×3排列 | 全链路、输出、断点、成本表 |
| 1 正确性 | 合成 n=20,000，2组×10排列 | 每次重新初始化；排列可复现；batch均摊正确；效率恒等式成立 |
| 2 归因检查 | 小数据，batch=1 与 batch=128 | 比较严格逐点结果和batch近似的Spearman相关性 |
| 3 规模基准 | 真实拆分，1组×1排列 | 实测训练、测试前向、GPU AUC、I/O及总时间，外推150次成本 |
| 4 正式运行 | 真实数据，3组×50排列 | 生成全部交付物 |

关键单元测试：

- 模型参数量、输出形状、Conv1d(k=1)与Linear等价性；
- FLOPs公式和MAC/FLOP口径；
- 分层拆分固定、训练/测试索引不相交且并集覆盖全部704,459行；
- 每次 permutation 都重新初始化模型和optimizer；
- 每次 permutation 使用不同但可复现的排列；
- batch贡献除以实际 batch 大小；
- `sum(marginal) ≈ final_auc - 0.5`；
- 组内50次取均值、组间3组计算 `ddof=1` 标准差；
- 全长输出的正式测试集和学习率验证集位置全部为 `NaN`；
- 中断续跑与无中断结果一致。

---

## 8. 实施顺序

1. 完成 config、seed派生、固定分层拆分、SGD学习率校准、模型、指标和合成数据模块。
2. 完成 FLOPs 公式、MAC校准和计时模块。
3. 实现单次 permutation：重新初始化、随机排列、batch均摊贡献、效率恒等式。
4. 实现每组50次循环、iteration级断点续跑和成本记录。
5. 实现3组调度及 mean/std、全长 NaN 对齐和 `3×50` 表汇总。
6. 依次完成冒烟、正确性测试和1次真实排列的规模基准。
7. 根据基准确认固定5,000点测试集逐batch精确GPU AUC是否可承受；如需近似，固定统一配置并写入报告。
8. 运行3组×50次正式实验，检查输出和恒等式后交付。

---

## 9. 已确认事项与实验限制

### 已确认

- 50指每组50次 G-Shapley permutation，不是同一模型训练50 epoch。
- 共3个独立组，因此总计150次排列。
- 每次排列重新初始化模型和optimizer，只做一次 one-pass epoch。
- optimizer固定为无动量SGD，学习率**已由同事校准完成，选定 0.1**，全部 150 次排列冻结使用（方法见 §2.2，存档 `lr_search.json`）。
- batch size固定为128，采用 batch AUC增量除以实际batch大小的逐点均摊近似。
- 从原始数据固定分层抽取5,000点作正式测试集，并另保留5,000点学习率验证集。
- 最终有效估值数量为 `n_train`；长度704,459的对齐数组中，两个保留集位置填 `NaN`。
- 初始AUC价值固定为0.5。

### 限制

1. batch=128只能得到mini-batch均摊近似，不是严格逐点 G-Shapley。
2. 只有3组，逐点标准差可以计算，但统计稳定性有限。
3. 缺少独立测试集和视频ID，随机片段拆分存在同视频数据泄漏风险。
4. 标签由路径是否包含 `Normal` 生成，可能是视频级弱标签复制到片段；最终报告需说明其标签语义。
5. 每个batch后重新计算AUC仍是主要计算成本（约为训练核心FLOPs的13倍），正式运行前必须完成一次真实排列基准。
6. 已核验PLOVAD官方源码：两层classifier均为`kernel_size=1, padding=0`；在`cls_hidden=128`时参数量为65,793。
7. PLOVAD原训练使用Adam(`lr=1e-3`)，但正式G-Shapley按论文参考实现改用无动量SGD；模型结构来自PLOVAD，优化规则来自G-Shapley。

---

## 10. 参考资料

- 论文：Ghorbani & Zou, *What is your data worth? Equitable Valuation of Data*, ICML 2019（本地 `1904.02868v2.pdf`；G-Shapley见Algorithm 2及附录B）。
- PLOVAD官方实现：[ctX-u/PLOVAD](https://github.com/ctX-u/PLOVAD)，[decoder源码](https://github.com/ctX-u/PLOVAD/blob/master/src/model.py#L109-L113)明确使用两层`kernel_size=1`；超参见 `src/configs_base2novel.py`。
- 本地参考实现：[DShap.py](DShap.py)、[Shapley.py](Shapley.py)、[shap_utils.py](shap_utils.py)。
