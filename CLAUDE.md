# DataShapley（G-Shapley 数据价值评估）

## 项目背景

用 G-Shapley（Ghorbani & Zou 2019，Algorithm 2）评估 UCF-Crime 704,459 个视频段的逐点数据价值。

- 正式实验：3 个独立组（master seeds `5922 / 13764 / 14933`，见 `seeds_used.json`）× 每组 50 次随机排列 = **150 次**
- 分类器 = PLOVAD decoder（2 层 `k=1` Conv1d + GELU，65,793 参数）；**无动量 SGD，学习率已校准冻结 = 0.1**（同事完成校准；不用 plovad 的 Adam 1e-3）
- batch=128；每 batch 训练后测 5,000 点测试集 AUC，增量按 batch 均摊（`delta/len(B)`）；`V(empty)=0.5`；**效率恒等式 `sum(marginal) ≈ final_auc − 0.5` 必须通过**
- 详细方案见 [G-Shapley实施计划.md](G-Shapley实施计划.md)（实验口径、FLOPs 测量、输出格式、验证步骤全部以它为准）

## 远程机器（每次操作远程前必读）

**在远程机器 `/data/ldy/DataShapley` 上工作前，先读取 [远程运行环境确认清单.md](远程运行环境确认清单.md)** —— 它是远程机器的标准说明文档，包含硬件/软件/路径/运行策略/开跑门槛状态。核心速查：

- **4× RTX 3090（24GB）**，driver 495.29.05（CUDA 11.5，装不了新 torch cu121，勿尝试升级驱动）；**⚠️ GPU 2 损坏（使用会死机，禁止使用），只可用 GPU 0/1/3**
- 默认环境：`/data/conda_envs/insight_env/bin/python`（py3.10 + torch 2.5.1+cu118，GPU 实测可用）；后备：base（torch 1.13.1+cu117）
- 数据：`/data/ldy/DataShapley/packed_data`（npy 不提交 git）；代码：`/data/ldy/DataShapley`
- 运行：3 组并行绑 GPU 0/1/2，直接 shell 无调度器；正式开跑前必须满足清单 §6 的全部门槛

## 约定

- `packed_data/`、`*.7z`、`.DS_Store` 不提交 git（已在 `.gitignore`）
- 种子：共享池 `seeds_100.json`（`RandomState(41)` 生成）；本仓库 3 组 = `seeds_used.json`；同事用 `seeds[3:6]`，不得重叠
- 代码在 `gshap_pytorch/`；测试 `pytest gshap_pytorch/tests`（远程需先装 pytest、thop）
- 远程 git 操作注意代理：git 走 `127.0.0.1:7897`（本机 Clash），pip 用清华源
