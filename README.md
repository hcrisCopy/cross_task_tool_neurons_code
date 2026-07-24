# Cross-Task Tool-Decision Neurons

用于规划和实现跨任务类型工具调用共享神经元实验的代码仓库。

## To-do List

- [ ] 检查 `tool_neurons` 环境和 `requirements.txt`
- [ ] 准备原始 When2Tool 数据集
- [ ] 准备 6 个本地大模型权重
- [ ] 阶段 1：原始数据准备
- [ ] 阶段 2：模型 0/1 标签生成
- [ ] 阶段 3：模型专属改造后数据集构建
- [ ] 阶段 4：特征和 FFN 激活提取
- [ ] 阶段 5：A/B/C 单类型神经元探测
- [ ] 阶段 6：跨任务类型共享神经元发现
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

待补充内容：

- [ ] 检查原始数据 split 是否齐全
- [ ] 检查 single-hop / multi-hop 样本数
- [ ] 检查 env 到 A/B/C 的映射
- [ ] 生成 raw data manifest

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

待补充内容：

- [ ] 标签生成入口脚本
- [ ] train/test 同时跑标签
- [ ] hard-no-tool 运行设置
- [ ] 每模型标签 manifest
- [ ] 断点续跑和完整性检查

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

待补充内容：

- [ ] 合并原始字段、A/B/C 类型、模型专属标签
- [ ] 保留原始 env/tool schema
- [ ] 保留 difficulty metadata
- [ ] 输出字段 schema 说明

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

待补充内容：

- [ ] current/no-reasoning prompt 构造
- [ ] Qwen XML / Llama native tool format 分流
- [ ] FFN target module hook
- [ ] last-token activation 保存
- [ ] 多卡切分和合并

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

待补充内容：

- [ ] A/B/C 分别探测
- [ ] SCAR 分数计算
- [ ] 每类型全模型 top-k
- [ ] 单类型神经元热力图

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

待补充内容：

- [ ] A/B/C top-k 精确交集
- [ ] pairwise overlap 统计
- [ ] CTD 热力图
- [ ] share rate 汇总

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
