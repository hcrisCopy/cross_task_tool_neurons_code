# ProbePrefill 单机八卡运行指令

本文档给出 PP-3、PP-4 和 PP-5 内置数据并行启动器的正式实验命令。参数与 `tmpREADME.md` 保持一致；仅将需要生成的阶段改为每张 GPU 启动一个独立模型进程。

PP-1 和 PP-2 不加载大型生成模型，因此保持单进程运行：

- PP-1 从已有的阶段 4 activation 文件构建 CTD probe 特征。
- PP-2 在 CPU 上使用原始 When2Tool 风格的求解器配置训练轻量 sklearn 逻辑回归 probe。

对于 PP-3 至 PP-5，传入 `--gpus 0,1,2,3,4,5,6,7` 后，每张 GPU 会启动一个 worker 进程。每个 worker 获取测试题不同的 `i::8` 切片，并在自己的 GPU 上加载一份完整模型。父进程显示全局终端进度条，随后将各分片的 `outputs.json`、`per_task.jsonl`、汇总、指标和 delta 文件合并回正常的最终输出目录。

ProbePrefill 评测阶段统一使用随机种子 `2026`。

```bash
cd cross_task_tool_neurons_code
conda activate tool_neurons
```

## PP-1：构建全量 CTD probe 特征

PP-1 不加载生成模型，因此保持单进程。正式实验使用全部可用的 train 和 test 样本（两个上限均为 `0`）：

```bash
python ProbePrefill/pp_build_probe_features.py \
  --model-alias qwen3-4b-instruct \
  --subset all \
  --max-train-samples 0 \
  --max-test-samples 0 \
  --sample-strategy balanced \
  --require-per-type-labels \
  --seed 2026
```

## PP-2：训练 CTD probe

PP-2 在 CPU 上训练轻量 sklearn 逻辑回归 probe，同样保持单进程：

```bash
python ProbePrefill/pp_train_probe.py \
  --model-alias qwen3-4b-instruct \
  --subset all \
  --reg 10000 \
  --max-iter 2000 \
  --threshold 0.5
```

## PP-3：Probe&Prefill 评测（八卡）

```bash
python ProbePrefill/pp_eval_probe_prefill.py \
  --model-alias qwen3-4b-instruct \
  --subset all \
  --thresholds 0.1,0.3,0.5,0.7,0.9 \
  --temperature 2.0 \
  --prefill-mode auto \
  --backend vllm \
  --n-runs 3 \
  --batch-size 1 \
  --max-rounds 10 \
  --max-new-tokens 2048 \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --vllm-dtype bfloat16 \
  --record-mode lite \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7
```

## PP-4：Base 评测与 delta 计算（八卡）

PP-4 必须在 PP-3 已生成最终 Probe&Prefill 输出后运行。

```bash
python ProbePrefill/pp_eval_base_and_delta.py \
  --model-alias qwen3-4b-instruct \
  --subset all \
  --thresholds 0.1,0.3,0.5,0.7,0.9 \
  --temperature 2.0 \
  --prefill-mode auto \
  --backend vllm \
  --n-runs 3 \
  --batch-size 1 \
  --max-rounds 10 \
  --max-new-tokens 2048 \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --vllm-dtype bfloat16 \
  --record-mode lite \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7
```

## PP-5：因果验证（八卡）

```bash
python ProbePrefill/pp_causal_validation.py \
  --model-alias qwen3-4b-instruct \
  --subset all \
  --reg 10000 \
  --max-iter 2000 \
  --threshold 0.5 \
  --interventions Base,Mask-Random,Mask-TDN_c,Mask-CTD,Mask-Private_c \
  --batch-size 1 \
  --max-rounds 10 \
  --max-new-tokens 2048 \
  --max-model-len 32768 \
  --torch-dtype bfloat16 \
  --device-map auto \
  --record-mode lite \
  --seed 2026 \
  --gpus 0,1,2,3,4,5,6,7
```

## 运行范围

以上命令是正式的八卡数据并行配置。PP-1 与 PP-2 不加载大型生成模型；PP-3、PP-4 以及 PP-5 的 activation-mask 部分均启动八个 worker，每个 worker 在分配到的 GPU 上加载一份完整模型。

## 与 When2Tool 对齐的生成设置

脚本保持与 When2Tool 对齐的评测设置：`prompt_mode=current`、`require_reasoning=False`、`enable_thinking=False`、`max_rounds=10`、`max_new_tokens=2048`、`max_model_len=32768`、`record_mode=lite`，且默认 `n_runs=3`。
