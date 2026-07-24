# Cross-Task Tool-Decision Neurons

用于规划和实现跨任务类型工具调用共享神经元实验的代码仓库。

## To-do List

- [x] 检查 `tool_neurons` 环境和 `requirements.txt`
- [x] 准备原始 When2Tool 数据集
- [x] 准备 qwen3-4b-instruct 本地权重
- [x] 阶段 1：原始数据准备
- [x] 阶段 2：模型 0/1 标签生成
- [x] 阶段 3：模型专属改造后数据集构建
- [x] 阶段 4：特征和 FFN 激活提取
- [x] 阶段 5：A/B/C 单类型神经元探测
- [x] 阶段 6：跨任务类型共享神经元发现
- [ ] 阶段 7：CTD-Masked LoRA 训练
- [ ] 阶段 8：训练后评测
- [ ] 阶段 9：因果验证
- [ ] 阶段 10：结果汇总和可视化
- [ ] 引用 When2Tool baseline 论文表格并整理最终对比表

## 章节目录

- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据和模型资源](#数据和模型资源)
- [阶段 1：原始数据准备](#阶段-1原始数据准备)
- [阶段 2：模型 0/1 标签生成](#阶段-2模型-01-标签生成)
- [阶段 3：改造后数据集构建](#阶段-3改造后数据集构建)
- [阶段 4：特征和激活提取](#阶段-4特征和激活提取)
- [阶段 5：单类型神经元探测](#阶段-5单类型神经元探测)
- [阶段 6：共享神经元发现](#阶段-6共享神经元发现)
- [阶段 7：CTD-Masked LoRA 训练](#阶段-7ctd-masked-lora-训练)
- [阶段 8：训练后评测](#阶段-8训练后评测)
- [阶段 9：因果验证](#阶段-9因果验证)
- [阶段 10：结果汇总和可视化](#阶段-10结果汇总和可视化)
- [命名规范](#命名规范)
- [代码和数据管理约定](#代码和数据管理约定)
- [参考链接](#参考链接)

## 目录结构

GitHub 只同步 `cross_task_tool_neurons_code/`。大文件、数据集、模型权重、训练 ckpt、评测输出、日志和图片都放在仓库同级的数据或模型目录中，不提交到 GitHub。

假设当前仓库路径是：

```text
.../cross_task_tool_neurons_code/
```

推荐同级目录关系如下：

```text
.../
|-- cross_task_tool_neurons_code/       # GitHub 同步的代码仓库
|   |-- README.md
|   |-- requirements.txt
|   |-- .gitignore
|   |-- configs/
|   |   |-- models.yaml              # 模型 alias、repo_id、本地权重路径
|   |   |-- paths.yaml               # data root、输出 root、缓存 root
|   |   |-- experiment.yaml          # 默认 subset、split、seed、top-k 等实验配置
|   |   |-- stage_defaults.yaml      # 各阶段默认参数
|   |-- code/
|       |-- 00_common/               # 公共 IO、路径、模型、hook、指标、画图工具
|       |-- 01_raw_data_preparation/ # 阶段 1：检查和整理原始 When2Tool 数据
|       |-- 02_labeling/             # 阶段 2：6 个模型分别跑 tool_necessary 标签
|       |-- 03_dataset_building/     # 阶段 3：构建模型专属改造后数据集
|       |-- 04_activation_extraction/# 阶段 4：抽取 FFN last-token activation
|       |-- 05_single_type_discovery/# 阶段 5：A/B/C 单类型神经元探测
|       |-- 06_shared_discovery/     # 阶段 6：A/B/C 交集共享神经元发现
|       |-- 07_training/             # 阶段 7：CTD-Masked LoRA 训练
|       |-- 08_evaluation/           # 阶段 8：训练后评测
|       |-- 09_causal_validation/    # 阶段 9：因果验证
|       |-- 10_reporting/            # 阶段 10：汇总表和可视化
|       |-- 11_multigpu/             # 单机多卡调度入口
|
|-- cross_task_tool_neurons_data/       # 不提交 GitHub
|   |-- datasets/                    # 原始和改造后 When2Tool 数据
|   |-- labels/                      # 阶段 2 输出：每个模型的 0/1 标签
|   |-- activations/                 # 阶段 4 输出：激活
|   |-- neurons/                     # 阶段 5/6 输出：单类型和共享神经元
|   |-- checkpoints/                 # 阶段 7 输出：LoRA checkpoint
|   |-- outputs/                     # 阶段 8/10 输出：评测和汇总结果
|   |-- causal_validation/           # 阶段 9 输出：因果验证结果
|   |-- visualizations/              # 热力图和结果图
|
|-- Qwen/                              # 不提交 GitHub
|   |-- qwen3-1.7b/
|   |-- qwen3-4b-instruct/
|   |-- qwen3-14b/
|   |-- qwen3-32b/
|
|-- meta-llama/                        # 不提交 GitHub
    |-- llama3.1-8b/
    |-- llama3.3-70b/
```

## 环境配置

环境名固定为 `tool_neurons`，`requirements.txt` 直接沿用参考仓库配置。

```bash
conda create -n tool_neurons python=3.10 -y
conda activate tool_neurons

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install modelscope
```

## 数据和模型资源

先进入仓库根目录：

```bash
cd .../cross_task_tool_neurons_code
mkdir -p ../cross_task_tool_neurons_data/datasets/raw_when2tool
```

下载原始 When2Tool 数据集：

```bash
huggingface-cli download cesun/When2Tool \
  --repo-type dataset \
  --local-dir ../cross_task_tool_neurons_data/datasets/raw_when2tool \
  --local-dir-use-symlinks False
```

下载模型权重。代码默认禁止自动从 Hugging Face 缓存下载，所有模型都要提前放到 `configs/models.yaml` 对应的本地路径。

```bash
mkdir -p ../Qwen ../meta-llama

modelscope download --model Qwen/Qwen3-1.7B --local_dir ../Qwen/qwen3-1.7b
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir ../Qwen/qwen3-4b-instruct
modelscope download --model Qwen/Qwen3-14B --local_dir ../Qwen/qwen3-14b
modelscope download --model Qwen/Qwen3-32B --local_dir ../Qwen/qwen3-32b

modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir ../meta-llama/llama3.1-8b
modelscope download --model LLM-Research/Llama-3.3-70B-Instruct --local_dir ../meta-llama/llama3.3-70b
```

后续阶段统一通过模型标签读取 `configs/models.yaml` 中的本地路径；切换模型时只改这一行：

```bash
MODEL_ALIAS=qwen3-4b-instruct
```

模型 alias 固定为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

本次小样本打通固定使用：

```text
MODEL_ALIAS=qwen3-4b-instruct
single-hop train/test = 100 / 30
multi-hop train/test = 40 / 30
prompt = hard_no_tool 标签阶段；current/no_reasoning 激活阶段
enable_thinking = false
top_k = 5000
```

本次远端打通结果摘要：

```text
single-hop labels: train 100 (y=0/1: 51/49), test 30 (y=0/1: 14/16)
multi-hop labels: train 40 (y=0/1: 21/19), test 30 (y=0/1: 15/15)
activation modules: 108 FFN modules = 36 layers * gate/up/down
single-hop CTD: 49 neurons, pairwise AB/AC/BC = 498/398/108
multi-hop CTD: 25 neurons, pairwise AB/AC/BC = 278/198/102
```

When2Tool 官方仓库不复制进本仓库；运行时放在同级 `../when2tool_repo`，用于导入官方 env、tool schema、prompt、state machine 和 vLLM/HF wrapper。若不存在，先运行：

```text
python code/00_common/sync_when2tool_repo.py --repo-dir ../when2tool_repo --network-turbo /etc/network_turbo --pull
```

## 阶段 1：原始数据准备

代码位置：

```text
code/01_raw_data_preparation/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/raw_when2tool/
```

该目录只放 When2Tool 原始数据，后续阶段 2 跑标签时直接读取这里，不读取改造后数据集。

本阶段只检查原始 parquet，不改写数据。

运行指令：

```text
python code/01_raw_data_preparation/inspect_raw_data.py --overwrite
```

输出：

```text
../cross_task_tool_neurons_data/datasets/raw_when2tool/manifest.json
```

做法：

- 检查 `single_hop` / `multi_hop` 的 `train` / `test` parquet 是否存在。
- 统计样本数、env、difficulty、A/B/C task_type 分布。
- A/B/C 映射来自实验方案：A=计算规模，B=知识边界，C=执行可靠性。

## 阶段 2：模型 0/1 标签生成

代码位置：

```text
code/02_labeling/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/raw_when2tool/
```

输出按模型隔离，禁止不同模型互相覆盖：

```text
../cross_task_tool_neurons_data/labels/<model_alias>/
|-- single_hop/
|   |-- train/labels.jsonl
|   |-- test/labels.jsonl
|-- multi_hop/
    |-- train/labels.jsonl
    |-- test/labels.jsonl
```

运行指令：

```text
python code/02_labeling/generate_tool_necessity_labels.py --model-alias qwen3-4b-instruct --when2tool-repo ../when2tool_repo --single-train-count 100 --single-test-count 30 --multi-train-count 40 --multi-test-count 30 --candidate-multiplier 2.0 --require-per-type-labels --backend vllm --tensor-parallel-size 1 --max-model-len 4096 --max-new-tokens 512 --max-rounds 8
```

如需清理旧的错误标签产物后重跑，在同一命令末尾加：

```text
--clean
```

输出：

```text
../cross_task_tool_neurons_data/labels/qwen3-4b-instruct/
|-- single_hop/train/labels.jsonl
|-- single_hop/train/candidate_labels.jsonl
|-- single_hop/train/raw_outputs.json
|-- single_hop/train/summary.json
|-- single_hop/train/manifest.json
|-- single_hop/test/...
|-- multi_hop/train/...
|-- multi_hop/test/...
```

做法：

- 用官方 When2Tool `AgentModel + evaluate_batched`，`backend=vllm`。
- 标签阶段固定 `hard_no_tool + no_reasoning + enable_thinking=false`。
- `hard_no_tool` 先使用 no-tool prompt，再在 state machine 里拒绝工具调用。
- 每个 split 先按 A/B/C 和 difficulty 抽候选样本，再按跑出的 `tool_necessary=0/1` 二次筛选目标数量。
- `tool_necessary = 1 - no_tool_correct`，只来自该模型 hard no-tool 结果。
- 已存在且 manifest 参数一致时提前跳过；`--clean` 会先删除该阶段对应模型/split 产物。

## 阶段 3：改造后数据集构建

代码位置：

```text
code/03_dataset_building/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/raw_when2tool/
../cross_task_tool_neurons_data/labels/<model_alias>/
```

输出：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/
|-- manifest.json
|-- single_hop/
|   |-- train.jsonl
|   |-- test.jsonl
|-- multi_hop/
    |-- train.jsonl
    |-- test.jsonl
```

运行指令：

```text
python code/03_dataset_building/build_modified_dataset.py --model-alias qwen3-4b-instruct
```

如需清理旧的错误改造数据集后重跑，在同一命令末尾加：

```text
--clean
```

输出：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/
|-- manifest.json
|-- single_hop/train.jsonl
|-- single_hop/test.jsonl
|-- multi_hop/train.jsonl
|-- multi_hop/test.jsonl
```

做法：

- 读取阶段 2 选出的 `labels.jsonl`，不重新生成标签。
- 合并原始 instruction、env、tool schema、parameters、expected answer、steps、difficulty、tags。
- 增加 `task_type`、`tool_necessary`、`no_tool_correct`、`label_metadata`。
- 单跳/多跳、train/test 均分开保存；不同模型写入不同 `<model_alias>` 目录。

## 阶段 4：特征和激活提取

代码位置：

```text
code/04_activation_extraction/
code/11_multigpu/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/
```

输出：

```text
../cross_task_tool_neurons_data/activations/<model_alias>/
|-- manifest.json
|-- single_hop/
|   |-- train/
|   |   |-- activations.pt
|   |   |-- meta.jsonl
|   |   |-- summary.json
|   |-- test/
|       |-- activations.pt
|       |-- meta.jsonl
|       |-- summary.json
|-- multi_hop/
    |-- train/
    |-- test/
```

运行指令：

```text
python code/04_activation_extraction/extract_ffn_activations.py --model-alias qwen3-4b-instruct --when2tool-repo ../when2tool_repo --batch-size 1
```

如需清理旧的错误激活产物后重跑，在同一命令末尾加：

```text
--clean
```

输出：

```text
../cross_task_tool_neurons_data/activations/qwen3-4b-instruct/
|-- single_hop/train/activations.pt
|-- single_hop/train/meta.jsonl
|-- single_hop/train/summary.json
|-- single_hop/test/...
|-- multi_hop/train/...
|-- multi_hop/test/...
```

做法：

- 激活 prompt 固定 `current + no_reasoning + enable_thinking=false`。
- 系统提示按模型分流：Qwen 使用 XML `<tool_call>`，Llama 使用 native tool calling。
- 每条样本只暴露自己所属 env 的 tool schema。
- hook 目标模块为 `mlp.gate_proj`、`mlp.up_proj`、`mlp.down_proj`。
- 每个样本只保存最后一个输入 token 的 FFN 模块输出坐标，不做 token 平均。
- `activations.pt` 保存 `{module_key: tensor[num_samples, dim]}`，`meta.jsonl` 保存 id、A/B/C、label、difficulty。

## 阶段 5：单类型神经元探测

代码位置：

```text
code/05_single_type_discovery/
code/11_multigpu/
```

输入：

```text
../cross_task_tool_neurons_data/activations/<model_alias>/
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/
```

输出：

```text
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/
|-- manifest.json
|-- single_hop/
|   |-- A/
|   |   |-- TDN_neurons.jsonl
|   |   |-- scar_scores.pt
|   |   |-- summary.json
|   |-- B/
|   |-- C/
|-- multi_hop/
    |-- A/
    |-- B/
    |-- C/
```

可视化输出：

```text
../cross_task_tool_neurons_data/visualizations/<model_alias>/single_type_by_subset/
|-- single_hop_heatmap.png
|-- multi_hop_heatmap.png
|-- summary.md
```

运行指令：

```text
python code/05_single_type_discovery/discover_single_type_neurons.py --model-alias qwen3-4b-instruct --top-k 5000 --min-class-count 2
```

如需清理旧的错误单类型神经元产物后重跑，在同一命令末尾加：

```text
--clean
```

输出：

```text
../cross_task_tool_neurons_data/neurons/qwen3-4b-instruct/single_type_by_subset/
|-- single_hop/A/TDN_neurons.jsonl
|-- single_hop/A/scar_scores.pt
|-- single_hop/A/summary.json
|-- single_hop/B/...
|-- single_hop/C/...
|-- multi_hop/A/...

../cross_task_tool_neurons_data/visualizations/qwen3-4b-instruct/single_type_by_subset/
|-- single_hop_heatmap.png
|-- multi_hop_heatmap.png
```

做法：

- 只使用 train split 的激活和标签；test 不参与神经元发现。
- A/B/C 分别探测：A 只用 A 类样本，B 只用 B 类样本，C 只用 C 类样本。
- `tool_necessary=1` 对应 Safety Kernel 的 unsafe 条件，`tool_necessary=0` 对应 safe 条件。
- SCAR 按实验方案实现：`delta=mu1-mu0`，pooled std 里加 `epsilon`，`D=z(delta/(rho+epsilon))`，`R=z(delta)`，`SCAR=D+R`。
- z-score 在同一 `(layer, module)` 内对全部输出坐标归一化。
- 每个类型从全模型候选 FFN 输出坐标中全局选 `top_k=5000`，不按层、不按模块分配名额。

## 阶段 6：共享神经元发现

代码位置：

```text
code/06_shared_discovery/
```

输入：

```text
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/
```

输出：

```text
../cross_task_tool_neurons_data/neurons/<model_alias>/shared_by_subset/
|-- manifest.json
|-- shared_summary.csv
|-- single_hop/
|   |-- CTD_neurons.jsonl
|   |-- pairwise_AB_neurons.jsonl
|   |-- pairwise_AC_neurons.jsonl
|   |-- pairwise_BC_neurons.jsonl
|   |-- layer_counts.csv
|   |-- module_counts.csv
|   |-- share_rates.csv
|-- multi_hop/
```

可视化输出：

```text
../cross_task_tool_neurons_data/visualizations/<model_alias>/shared_by_subset/
|-- shared_neuron_heatmap_single_hop.png
|-- shared_neuron_heatmap_multi_hop.png
|-- summary.md
```

运行指令：

```text
python code/06_shared_discovery/discover_shared_neurons.py --model-alias qwen3-4b-instruct
```

如需清理旧的错误共享神经元产物后重跑，在同一命令末尾加：

```text
--clean
```

输出：

```text
../cross_task_tool_neurons_data/neurons/qwen3-4b-instruct/shared_by_subset/
|-- manifest.json
|-- shared_summary.csv
|-- single_hop/CTD_neurons.jsonl
|-- single_hop/pairwise_AB_neurons.jsonl
|-- single_hop/pairwise_AC_neurons.jsonl
|-- single_hop/pairwise_BC_neurons.jsonl
|-- single_hop/share_rates.csv
|-- multi_hop/...

../cross_task_tool_neurons_data/visualizations/qwen3-4b-instruct/shared_by_subset/
|-- shared_neuron_heatmap_single_hop.png
|-- shared_neuron_heatmap_multi_hop.png
```

做法：

- 读取阶段 5 的 `TDN_neurons.jsonl`。
- 按完整神经元身份 `(layer, module, index)` 精确匹配。
- `CTD = TDN_A ∩ TDN_B ∩ TDN_C`。
- 额外保存 `AB/AC/BC` pairwise overlap。
- `share_rate = |CTD| / |TDN_c|`，分别对 A/B/C 汇总。

## 阶段 7：CTD-Masked LoRA 训练

代码位置：

```text
code/07_training/
code/11_multigpu/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/
../cross_task_tool_neurons_data/neurons/<model_alias>/shared_by_subset/
```

输出：

```text
../cross_task_tool_neurons_data/checkpoints/<model_alias>/ctd_masked_lora/
|-- single_hop/
|   |-- adapter/
|   |-- training_log.csv
|   |-- training_examples.jsonl
|   |-- skipped_examples.jsonl
|   |-- mask_summary.json
|   |-- manifest.json
|-- multi_hop/
```

待补充内容：

- [ ] 训练轨迹构造
- [ ] CTD mask 构造
- [ ] LoRA adapter 注入
- [ ] assistant token loss mask
- [ ] checkpoint manifest

## 阶段 8：训练后评测

代码位置：

```text
code/08_evaluation/
code/11_multigpu/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/
../cross_task_tool_neurons_data/checkpoints/<model_alias>/ctd_masked_lora/
```

输出：

```text
../cross_task_tool_neurons_data/outputs/<model_alias>/trained_evaluation/
|-- manifest.json
|-- single_hop/
|   |-- outputs.json
|   |-- per_task.jsonl
|   |-- summary.json
|-- multi_hop/
```

待补充内容：

- [ ] 加载 CTD-Masked LoRA adapter
- [ ] current/no-reasoning test split 评测
- [ ] When2Tool 指标聚合
- [ ] 按 overall / env / difficulty / A-B-C 分组

## 阶段 9：因果验证

代码位置：

```text
code/09_causal_validation/
code/11_multigpu/
```

输入：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/
../cross_task_tool_neurons_data/neurons/<model_alias>/
```

输出：

```text
../cross_task_tool_neurons_data/causal_validation/<model_alias>/
|-- manifest.json
|-- single_hop/
|   |-- summary_table.csv
|   |-- A/
|   |   |-- Base/
|   |   |-- Mask-CTD/
|   |-- B/
|   |-- C/
|-- multi_hop/
```

待补充内容：

- [ ] Base vs Mask-CTD 因果验证
- [ ] mask hook
- [ ] 工具行为指标
- [ ] 因果验证汇总表

## 阶段 10：结果汇总和可视化

代码位置：

```text
code/10_reporting/
```

输入：

```text
../cross_task_tool_neurons_data/labels/<model_alias>/
../cross_task_tool_neurons_data/neurons/<model_alias>/
../cross_task_tool_neurons_data/checkpoints/<model_alias>/ctd_masked_lora/
../cross_task_tool_neurons_data/outputs/<model_alias>/trained_evaluation/
../cross_task_tool_neurons_data/causal_validation/<model_alias>/
```

输出：

```text
../cross_task_tool_neurons_data/outputs/final_report/
|-- model_summary.csv
|-- training_comparison.csv
|-- causal_validation_summary.csv
|-- neuron_discovery_summary.csv
|-- figures/
|-- README_results.md
```

本实验暂不复现 When2Tool baseline。`Default` / `Sparse` / `Reason-then-Act` / `Probe&Prefill` 结果直接引用 When2Tool 论文表格。本实验只新增运行 `CTD-Masked-LoRA`，并在相同模型、数据、prompt 设置下报告结果。

待补充内容：

- [ ] 汇总 6 个模型结果
- [ ] 汇总 single-hop / multi-hop 结果
- [ ] 引用 When2Tool baseline 表格
- [ ] 生成最终对比表
- [ ] 生成最终图

## 命名规范

所有阶段输出都必须包含 `<model_alias>`，不同模型不能相互覆盖。

固定枚举：

```text
<model_alias> = qwen3-1.7b | qwen3-4b-instruct | qwen3-14b | qwen3-32b | llama3.1-8b | llama3.3-70b
<subset>      = single_hop | multi_hop
<split>       = train | test
<task_type>   = A | B | C
```

推荐输出路径模板：

```text
../cross_task_tool_neurons_data/<stage_output>/<model_alias>/<subset>/<split_or_task_type>/
```

每个正式输出目录建议至少包含：

```text
manifest.json
summary.json 或 summary.md
```

## 代码和数据管理约定

- `cross_task_tool_neurons_code/` 是唯一 GitHub 仓库。
- `cross_task_tool_neurons_data/`、`Qwen/`、`meta-llama/` 不提交 GitHub。
- 不要把完整数据集、模型权重、LoRA checkpoint、`.pt` 激活、日志、图片批量提交到代码仓库。
- 代码默认使用相对路径，例如 `../cross_task_tool_neurons_data`、`../Qwen`、`../meta-llama`。
- 不要硬编码个人电脑路径、服务器用户名、私有挂载点、token、cookie、secret。
- 缺数据、缺权重、路径错误时直接报错，不要静默跳过，也不要自动换路径。
- 每个阶段都要支持按 `<model_alias>` 独立运行。
- 每个阶段输出都要写入模型专属目录，避免 6 个模型结果互相覆盖。
- 默认不接入 wandb 等外部平台，除非后续明确需要。

## 参考链接

- Safety Kernel / Who Transfers Safety?: https://arxiv.org/abs/2602.01283
- When2Tool paper: https://arxiv.org/abs/2605.09252
- When2Tool arXiv: https://arxiv.org/abs/2605.09252
- When2Tool code: https://github.com/Trustworthy-ML-Lab/when2tool
- When2Tool dataset: https://huggingface.co/datasets/cesun/When2Tool
- Qwen3 models: https://modelscope.cn/organization/Qwen
- Llama ModelScope mirrors: https://modelscope.cn/organization/LLM-Research
