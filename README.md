# Cross-Task Tool-Decision Neurons

基于 When2Tool 和 Who Transfers Safety? / Safety Kernel 方法，研究 A/B/C 三类任务中影响“是否调用工具”的单类型工具决策神经元与跨类型共享神经元。

## To-do List

- [x] 内置官方 When2Tool 代码到 `third_party/when2tool/`
- [ ] 准备原始 When2Tool 数据集和本地模型权重
- [ ] 对 6 个模型跑模型专属 `tool_necessary` 标签
- [ ] 基于标签构建模型专属改造后数据集
- [ ] 抽取 train/test FFN last-token activation
- [ ] 在 train split 上探测 A/B/C 单类型工具决策神经元
- [ ] 发现 A/B/C 跨类型共享神经元
- [ ] 训练 CTD-Masked LoRA
- [ ] 在 test split 上评测训练后模型
- [ ] 在 test split 上评测同口径 Base/Default 并计算 delta
- [ ] 在 test split 上做因果验证
- [ ] 汇总 6 个模型最终结果

## 章节目录

- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据和模型资源](#数据和模型资源)
- [核心约定](#核心约定)
- [阶段 1：原始数据检查](#阶段-1原始数据检查)
- [阶段 2：模型专属 tool_necessary 标签](#阶段-2模型专属-tool_necessary-标签)
- [阶段 3：构造模型专属改造数据集](#阶段-3构造模型专属改造数据集)
- [阶段 4：FFN last-token activation 提取](#阶段-4ffn-last-token-activation-提取)
- [阶段 5：A/B/C 单类型神经元发现](#阶段-5abc-单类型神经元发现)
- [阶段 6：跨任务类型共享神经元](#阶段-6跨任务类型共享神经元)
- [阶段 7：CTD-Masked LoRA 训练](#阶段-7ctd-masked-lora-训练)
- [阶段 8：训练后评测](#阶段-8训练后评测)
- [阶段 9：Base 模型评测与 delta 计算](#阶段-9base-模型评测与-delta-计算)
- [阶段 10：因果验证](#阶段-10因果验证)
- [阶段 11：结果汇总](#阶段-11结果汇总)
- [清理旧错误产物](#清理旧错误产物)
- [六模型运行顺序](#六模型运行顺序)
- [参考](#参考)

## 目录结构

假设当前仓库路径是：

```text
.../cross_task_tool_neurons_code/
```

大文件、输出、模型权重不提交到 GitHub，放在代码仓库同级目录。官方 When2Tool 代码是轻量源码依赖，已经放在本仓库 `third_party/when2tool/` 并随 GitHub 同步。

```text
.../
|-- cross_task_tool_neurons_code/       # GitHub 同步的代码仓库
|   |-- README.md
|   |-- requirements.txt
|   |-- configs/
|   |   |-- models.yaml                # 模型 alias、repo_id、本地权重路径
|   |   |-- paths.yaml                 # data root、输出 root、When2Tool 代码路径
|   |   |-- experiment.yaml            # 默认 subset、split、seed、top-k 等实验配置
|   |   |-- stage_defaults.yaml        # 各阶段默认参数
|   |-- code/
|   |   |-- 00_common/                 # 公共 IO、路径、模型、hook、指标、画图工具
|   |   |-- 01_raw_data_preparation/   # 阶段 1：检查原始 When2Tool 数据
|   |   |-- 02_labeling/               # 阶段 2：生成模型专属 0/1 标签
|   |   |-- 03_dataset_building/       # 阶段 3：构建模型专属改造后数据集
|   |   |-- 04_activation_extraction/  # 阶段 4：抽取 FFN last-token activation
|   |   |-- 05_single_type_discovery/  # 阶段 5：A/B/C 单类型神经元发现
|   |   |-- 06_shared_discovery/       # 阶段 6：A/B/C 交集共享神经元发现
|   |   |-- 07_training/               # 阶段 7：CTD-Masked LoRA 训练
|   |   |-- 08_evaluation/             # 阶段 8/9：训练后评测、Base 评测与 delta
|   |   |-- 09_causal_validation/      # 阶段 10：因果验证
|   |   |-- 10_reporting/              # 阶段 11：汇总表和可视化
|   |   |-- 11_multigpu/               # 单机多卡 Python 调度入口
|   |-- third_party/
|       |-- when2tool/                 # 官方 When2Tool 代码，已随本仓库提交
|
|-- cross_task_tool_neurons_data/       # 不提交 GitHub
|   |-- datasets/
|   |   |-- raw_when2tool/             # 原始 When2Tool 数据集
|   |   |-- modified_when2tool/        # 阶段 3 输出：每个模型一份改造后数据集
|   |-- labels/                       # 阶段 2 输出：每个模型的 0/1 标签
|   |-- activations/                  # 阶段 4 输出：FFN activation
|   |-- neurons/                      # 阶段 5/6 输出：单类型和共享神经元
|   |-- checkpoints/                  # 阶段 7 输出：LoRA checkpoint
|   |-- outputs/                      # 阶段 8/9/11 输出：评测、Base delta 和汇总结果
|   |-- causal_validation/            # 阶段 10 输出：因果验证结果
|   |-- visualizations/               # 阶段 5/6/11 图
|
|-- Qwen/                             # 不提交 GitHub
|   |-- qwen3-1.7b/
|   |-- qwen3-4b-instruct/
|   |-- qwen3-14b/
|   |-- qwen3-32b/
|
|-- meta-llama/                       # 不提交 GitHub
    |-- llama3.1-8b/
    |-- llama3.3-70b/
```

## 环境配置

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

后续阶段统一通过模型标签读取 `configs/models.yaml` 中的本地路径；切换模型时只改命令中的：

```bash
--model-alias qwen3-4b-instruct
```

## 核心约定

- 固定模型标签：`qwen3-1.7b`、`qwen3-4b-instruct`、`qwen3-14b`、`qwen3-32b`、`llama3.1-8b`、`llama3.3-70b`。
- 下面阶段 2 到阶段 11 的命令都是单机八卡正式实验命令；每次只替换 `--model-alias qwen3-4b-instruct`。
- 单跳 `single_hop` 和多跳 `multi_hop` 全程分开；阶段 7 分别训练两个 adapter。
- 跑标签和构造数据使用 train+test；神经元发现和训练只使用 train；训练后评测、Base 评测和因果验证只使用 test。
- Qwen 走 XML tool call，Llama 走 native tool calling；prompt、tool schema、parser、state machine 复用 `third_party/when2tool` 官方代码。
- 正式实验样本数：single-hop train/test = `900/2250`，multi-hop train/test = `180/450`。
- 正式评测 `--n-runs 3`；smoke/debug 才手动改成 `--n-runs 1` 或减少样本数。
- 所有阶段都有参数敏感 manifest；产物存在且 manifest 一致会提前跳过，错误旧产物可在同一命令末尾加 `--clean` 清理重跑。

## 阶段 1：原始数据检查

检查 When2Tool 原始 parquet 的 split、样本数、env、difficulty 和 A/B/C 映射，不改写数据。

```text
python code/01_raw_data_preparation/inspect_raw_data.py --overwrite
```

输出：`../cross_task_tool_neurons_data/datasets/raw_when2tool/manifest.json`

## 阶段 2：模型专属 tool_necessary 标签

用官方 `AgentModel + evaluate_batched` 跑 `hard_no_tool/no_reasoning`。`tool_necessary = 1 - no_tool_correct`，标签只来自当前模型自己的无工具结果。

```text
python code/02_labeling/generate_tool_necessity_labels.py --model-alias qwen3-4b-instruct --raw-dataset-dir ../cross_task_tool_neurons_data/datasets/raw_when2tool --labels-dir ../cross_task_tool_neurons_data/labels --when2tool-repo third_party/when2tool --single-train-count 900 --single-test-count 2250 --multi-train-count 180 --multi-test-count 450 --candidate-multiplier 2.0 --require-per-type-labels --backend vllm --tensor-parallel-size 8 --max-model-len 32768 --max-new-tokens 2048 --max-rounds 12 --record-mode lite --vllm-dtype bfloat16
```

输出：`../cross_task_tool_neurons_data/labels/<model_alias>/<subset>/<split>/labels.jsonl`

## 阶段 3：构造模型专属改造数据集

合并原始 instruction/env/tool schema/expected/difficulty 与阶段 2 的模型专属标签，不重新跑模型。该阶段在八卡机器上运行，但不占 GPU。

```text
python code/03_dataset_building/build_modified_dataset.py --model-alias qwen3-4b-instruct --raw-dataset-dir ../cross_task_tool_neurons_data/datasets/raw_when2tool --labels-dir ../cross_task_tool_neurons_data/labels --output-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool
```

输出：`../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/<subset>/<split>.jsonl`

## 阶段 4：FFN last-token activation 提取

`auto` 对小/中模型使用数据 8 分片并自动合并；对 `qwen3-32b` 和 `llama3.3-70b` 默认使用模型并行 `device_map=auto`。

```text
python code/11_multigpu/run_activation_extraction.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0,1,2,3,4,5,6,7 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

输出：`../cross_task_tool_neurons_data/activations/<model_alias>/<subset>/<split>/activations.pt`

## 阶段 5：A/B/C 单类型神经元发现

只读 train activation。A/B/C 分别计算 `tool_necessary=1` vs `0` 的 SCAR，按全模型 FFN 输出坐标全局取 `top_k=5000`。

```text
python code/05_single_type_discovery/discover_single_type_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/activations --neurons-dir ../cross_task_tool_neurons_data/neurons --visualizations-dir ../cross_task_tool_neurons_data/visualizations --subset all --top-k 5000 --heatmap-top-n 300 --epsilon 1.0e-8 --min-class-count 2 --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
```

输出：`../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/TDN_neurons.jsonl`

## 阶段 6：跨任务类型共享神经元

按完整身份 `(layer, module, index)` 精确取交集：`CTD = TDN_A ∩ TDN_B ∩ TDN_C`，并输出 pairwise overlap 和热力图。

```text
python code/06_shared_discovery/discover_shared_neurons.py --model-alias qwen3-4b-instruct --neurons-dir ../cross_task_tool_neurons_data/neurons --visualizations-dir ../cross_task_tool_neurons_data/visualizations --subset all --heatmap-top-n 300
```

输出：`../cross_task_tool_neurons_data/neurons/<model_alias>/shared_by_subset/<subset>/CTD_neurons.jsonl`

## 阶段 7：CTD-Masked LoRA 训练

只用 train split。`tool_necessary=0` 用 hard-no-tool direct answer 轨迹；`tool_necessary=1` 用 `current/no_reasoning` 工具成功且最终答案正确的轨迹。loss 只算 assistant token，LoRA 更新只作用在 CTD mask 为 1 的 FFN 输出坐标。

```text
python code/11_multigpu/run_training.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --neurons-dir ../cross_task_tool_neurons_data/neurons --checkpoints-dir ../cross_task_tool_neurons_data/checkpoints --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --max-train-samples 0 --rank 8 --lora-alpha 16 --lora-dropout 0 --epochs 3 --per-device-batch-size 1 --gradient-accumulation-steps 16 --learning-rate 5e-5 --warmup-ratio 0.03 --max-grad-norm 1.0 --max-seq-length 4096 --trajectory-attempts 2 --trajectory-batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode full
```

输出：`../cross_task_tool_neurons_data/checkpoints/<model_alias>/ctd_masked_lora/<subset>/adapter/`

## 阶段 8：训练后评测

只用 test split，加载阶段 7 adapter，prompt 固定 `current/no_reasoning/enable_thinking=false`，不做 Probe&Prefill。本阶段只输出 `CTD-Masked-LoRA` 绝对指标；相对 Base 的 delta 在阶段 9 统一计算。

```text
python code/11_multigpu/run_evaluation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --checkpoints-dir ../cross_task_tool_neurons_data/checkpoints --outputs-dir ../cross_task_tool_neurons_data/outputs --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --max-test-samples 0 --n-runs 3 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite
```

输出：`../cross_task_tool_neurons_data/outputs/<model_alias>/trained_evaluation/<subset>/summary_table.csv`

## 阶段 9：Base 模型评测与 delta 计算

只用 test split，不加载 adapter，不做 activation mask，不做 Probe&Prefill。该阶段使用和阶段 8 相同的 HF/When2Tool 评测路径，避免 Base vs CTD-Masked-LoRA 的 delta 混入生成后端差异。

```text
python code/11_multigpu/run_base_evaluation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --outputs-dir ../cross_task_tool_neurons_data/outputs --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --max-test-samples 0 --n-runs 3 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite
```

输出：

```text
../cross_task_tool_neurons_data/outputs/<model_alias>/base_evaluation/<subset>/summary_table.csv
../cross_task_tool_neurons_data/outputs/<model_alias>/trained_evaluation/<subset>/comparison_with_base.csv
```

## 阶段 10：因果验证

只用 test split 和未训练 base 模型。比较 `Base`、`Mask-Random`、`Mask-TDN_c`、`Mask-CTD`、`Mask-Private_c`；mask 作用在 FFN 目标模块输出坐标，并覆盖所有 token 位置。

```text
python code/11_multigpu/run_causal_validation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --neurons-dir ../cross_task_tool_neurons_data/neurons --causal-dir ../cross_task_tool_neurons_data/causal_validation --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --max-test-samples 0 --interventions Base,Mask-Random,Mask-TDN_c,Mask-CTD,Mask-Private_c --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 20260725
```

输出：`../cross_task_tool_neurons_data/causal_validation/<model_alias>/<subset>/summary_table.csv`

## 阶段 11：结果汇总

只读已有产物，不重新跑模型。单模型汇总：

```text
python code/10_reporting/build_final_report.py --model-alias qwen3-4b-instruct --labels-dir ../cross_task_tool_neurons_data/labels --neurons-dir ../cross_task_tool_neurons_data/neurons --checkpoints-dir ../cross_task_tool_neurons_data/checkpoints --outputs-dir ../cross_task_tool_neurons_data/outputs --causal-dir ../cross_task_tool_neurons_data/causal_validation
```

六模型全部汇总：

```text
python code/10_reporting/build_final_report.py --model-alias all --labels-dir ../cross_task_tool_neurons_data/labels --neurons-dir ../cross_task_tool_neurons_data/neurons --checkpoints-dir ../cross_task_tool_neurons_data/checkpoints --outputs-dir ../cross_task_tool_neurons_data/outputs --causal-dir ../cross_task_tool_neurons_data/causal_validation
```

关键表格：

```text
base_evaluation_summary.csv
trained_evaluation_summary.csv
training_comparison.csv
causal_validation_summary.csv
causal_cross_type_summary.csv
model_summary.csv
```

## 清理旧错误产物

任一阶段发现旧产物错误，在原命令末尾追加：

```text
--clean
```

阶段 4 八卡入口的 `--clean` 会清理目标分片和合并后的正式 activation 产物。其他阶段会清理当前模型/当前阶段对应输出目录。所有清理都限制在 `../cross_task_tool_neurons_data` 内部。

## 六模型运行顺序

对每个模型依次把上述阶段 2 到阶段 11 命令里的 `--model-alias qwen3-4b-instruct` 替换为目标标签运行：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

跑完六个模型后执行阶段 11 的 `--model-alias all`。

## 参考

- Safety Kernel / Who Transfers Safety?: https://arxiv.org/abs/2602.01283
- When2Tool paper: https://arxiv.org/abs/2605.09252
- When2Tool code: https://github.com/Trustworthy-ML-Lab/when2tool
- When2Tool dataset: https://huggingface.co/datasets/cesun/When2Tool
- Qwen3 models: https://modelscope.cn/organization/Qwen
- Llama ModelScope mirrors: https://modelscope.cn/organization/LLM-Research
