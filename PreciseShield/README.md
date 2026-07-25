# PreciseShield 版阶段 4-11

本目录只放 PreciseShield 新方案代码。总 README 的阶段 1、阶段 2、阶段 3 保持不变：先生成每个模型自己的 `tool_necessary` 二分类标签，再构造模型专属改造后数据集。这里从阶段 4 接上，所有新产物都写到：

```text
../cross_task_tool_neurons_data/precise_shield/
```

不会覆盖旧方案的 `activations/neurons/checkpoints/outputs/causal_validation`。

## 目录结构

```text
PreciseShield/
|-- README.md
|-- plan.md
|-- ps_common.py
|-- ps_run_activation_extraction.py          # PS-4 八卡入口
|-- ps_extract_intermediate_activations.py   # PS-4 抽 FFN intermediate h
|-- ps_discover_single_type_neurons.py       # PS-5 A/B/C PS-TDN
|-- ps_discover_shared_neurons.py            # PS-6 PS-CTD
|-- ps_run_training.py                       # PS-7 八卡入口
|-- ps_train_masked_lora.py                  # PS-7 PreciseShield-Masked-LoRA
|-- ps_run_evaluation.py                     # PS-8 八卡入口
|-- ps_evaluate_trained_model.py             # PS-8 训练后评测
|-- ps_run_base_evaluation.py                # PS-9 八卡入口
|-- ps_evaluate_base_and_delta.py            # PS-9 Base 和 delta
|-- ps_run_causal_validation.py              # PS-10 八卡入口
|-- ps_causal_validation.py                  # PS-10 因果验证
|-- ps_build_final_report.py                 # PS-11 汇总
```

## 核心约定

模型标签固定为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

- 切换模型时只替换命令里的 `--model-alias qwen3-4b-instruct`。
- 单跳 `single_hop` 和多跳 `multi_hop` 全程分开保存；PS-7 会训练两个 adapter。
- 阶段 4 抽取 train/test；阶段 5/6 只用 train；阶段 7 只用 train；阶段 8/9/10 只用 test。
- Prompt、tool schema、parser、state machine 复用 `third_party/when2tool`，固定 `current/no_reasoning/enable_thinking=false`。
- Qwen 走 XML tool call，Llama 走 native tool calling，规则由公共代码按模型标签识别。
- 正式评测使用 `--n-runs 3`；小样本 smoke 才改成 `--n-runs 1`。
- 每阶段都有参数敏感 `manifest.json`。产物存在且参数一致会提前跳过；发现旧错误产物时在原命令末尾加 `--clean`，清理范围限制在 `../cross_task_tool_neurons_data/precise_shield/` 对应阶段内。

八卡策略：

- PS-4：按单机 8 张 H20 96GB 设计，`auto` 下 `qwen3-1.7b/qwen3-4b-instruct/qwen3-14b/llama3.1-8b` 按数据切 8 份；`qwen3-32b/llama3.3-70b` 用 `device_map=auto` 模型并行。
- PS-7/8/9/10：`auto` 下 `qwen3-32b/llama3.3-70b` 默认模型并行；其余模型在 `subset=all` 时默认 single-hop 和 multi-hop 并行跑，产物仍按 subset 隔离。
- PS-5/6 是已保存 activation 的统计和集合运算，不加载大模型。

## 方法定义

PreciseShield 神经元是 FFN intermediate：

```text
h = sigma(W_gate x + b_gate) * (W_up x + b_up)
n_i^(l) = h_i^(l)
```

代码通过 `down_proj` 的 `forward_pre_hook` 抓取 `down_proj` 输入处的 `h`，取模型开始生成前最后一个输入 token。

对样本集合 `D`：

```text
I_i^(l)(D) = |mean_D h_i^(l)| * ||W_down^(l)[:, i]||_2
S_i^(l)(D) = I_i^(l)(D) / (sum_j I_j^(l)(D) + epsilon)
```

每层取：

```text
k_l = max(floor(p * d_m), min_neurons_per_layer)
```

正式实验默认 `p=0.01`。

A/B/C 单类型：

```text
D_call,c   = task_type=c 且 tool_necessary=1
D_direct,c = task_type=c 且 tool_necessary=0
PS_TDN_c^(l) = TopK_l(S(D_call,c)) \ TopK_l(S(D_direct,c))
```

跨类型共享：

```text
PS_CTD = PS_TDN_A ∩ PS_TDN_B ∩ PS_TDN_C
```

训练只使用 `PS_CTD`。因果验证额外使用 `PS_TDN_c`、`PS_CTD`、`PS_TDN_c \ PS_CTD` 和同层同数量随机神经元。

## 前置阶段 1-3

先按总 README 跑完阶段 1、2、3。正式六模型阶段 2 样本数为：

```text
single-hop train/test = 900 / 2250
multi-hop train/test  = 180 / 450
```

阶段 3 输出供本目录读取：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/<subset>/<split>.jsonl
```

## PS-4：抽取 FFN intermediate 激活

```text
python PreciseShield/ps_run_activation_extraction.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/precise_shield/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0,1,2,3,4,5,6,7 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 20260725
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/summary.json
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/manifest.json
```

做法：用 When2Tool `init_state` 构造 prompt，hook 每层 `down_proj` 输入，保存最后一个输入 token 的 FFN intermediate `h`。`down_weight_norms` 同时保存给阶段 5 计算 PreciseShield saliency。

## PS-5：A/B/C 单类型神经元

```text
python PreciseShield/ps_discover_single_type_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/precise_shield/activations --neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield/visualizations --subset all --intervention-ratio 0.01 --min-neurons-per-layer 1 --heatmap-top-n 300 --epsilon 1.0e-12 --min-class-count 2
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/PS_TDN_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/saliency_scores.pt
../cross_task_tool_neurons_data/precise_shield/visualizations/<model_alias>/single_type_by_subset/*.png
```

做法：只读 train activation。每个 task type 内分别计算 `D_call` 和 `D_direct`，逐层 `TopK(S_call) \ TopK(S_direct)`。如果某类型的 0/1 标签数不足，直接报错，不改公式。

## PS-6：跨类型共享神经元

```text
python PreciseShield/ps_discover_shared_neurons.py --model-alias qwen3-4b-instruct --neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield/visualizations --subset all --heatmap-top-n 300
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/shared_by_subset/<subset>/PS_CTD_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/shared_by_subset/<subset>/pairwise_AB_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/shared_by_subset/<subset>/private_A_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/visualizations/<model_alias>/shared_by_subset/*.png
```

做法：按完整身份 `(layer, index)` 取交集，`PS_CTD = A ∩ B ∩ C`。pairwise 和 private 只用于分析和因果对照。

## PS-7：PreciseShield-Masked-LoRA 训练

```text
python PreciseShield/ps_run_training.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --checkpoints-dir ../cross_task_tool_neurons_data/precise_shield/checkpoints --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --parallel-mode auto --max-train-samples 0 --sample-strategy first --rank 8 --lora-alpha 16 --lora-dropout 0 --epochs 3 --per-device-batch-size 1 --gradient-accumulation-steps 16 --learning-rate 5e-5 --warmup-ratio 0.03 --max-grad-norm 1.0 --max-seq-length 4096 --trajectory-attempts 2 --trajectory-batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode full --seed 20260725
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/checkpoints/<model_alias>/ps_masked_lora/<subset>/adapter/
../cross_task_tool_neurons_data/precise_shield/checkpoints/<model_alias>/ps_masked_lora/<subset>/training_examples.jsonl
../cross_task_tool_neurons_data/precise_shield/checkpoints/<model_alias>/ps_masked_lora/<subset>/skipped_examples.jsonl
../cross_task_tool_neurons_data/precise_shield/checkpoints/<model_alias>/ps_masked_lora/<subset>/training_log.csv
../cross_task_tool_neurons_data/precise_shield/checkpoints/<model_alias>/ps_masked_lora/<subset>/manifest.json
```

做法：只用 train split。训练目标是第一步“调不调工具”的二分类行为。`tool_necessary=0` 监督直接 boxed answer；`tool_necessary=1` 只保留当前模型第一步成功 tool call 的样本。LoRA 只加在 `gate_proj/up_proj` 的 PS-CTD 行：

```text
W' = W0 + (M * B) A
```

不训练 `down_proj`、attention、embedding 和 base 权重。

## PS-8：训练后评测

```text
python PreciseShield/ps_run_evaluation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --checkpoints-dir ../cross_task_tool_neurons_data/precise_shield/checkpoints --outputs-dir ../cross_task_tool_neurons_data/precise_shield/outputs --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --parallel-mode auto --max-test-samples 0 --sample-strategy first --n-runs 3 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 20260725
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/outputs.json
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/per_task.jsonl
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/summary.json
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/summary_table.csv
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/manifest.json
```

做法：只用 test split，加载 PS-7 对应 subset adapter，按 When2Tool 官方流程评测。这里只写训练后绝对指标，Base delta 放到 PS-9。

## PS-9：Base 评测与 delta

```text
python PreciseShield/ps_run_base_evaluation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --outputs-dir ../cross_task_tool_neurons_data/precise_shield/outputs --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --parallel-mode auto --max-test-samples 0 --sample-strategy first --n-runs 3 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 20260725
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/base_evaluation/<subset>/summary.json
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/base_evaluation/<subset>/summary_table.csv
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/comparison_with_base.csv
../cross_task_tool_neurons_data/precise_shield/outputs/<model_alias>/trained_evaluation/<subset>/comparison_with_base_manifest.json
```

做法：Base 不加载 adapter，不做 activation mask。读取 PS-8 和本阶段 summary，输出 `delta_acc_pp`、`delta_avg_tool_calls`、`delta_tool_call_rate`、`tool_call_reduction_percent` 以及工具决策二分类 delta。

## PS-10：因果验证

```text
python PreciseShield/ps_run_causal_validation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --causal-dir ../cross_task_tool_neurons_data/precise_shield/causal_validation --when2tool-repo third_party/when2tool --subset all --gpus 0,1,2,3,4,5,6,7 --parallel-mode auto --max-test-samples 0 --sample-strategy first --interventions Base,Mask-Random,Mask-PS-TDN_c,Mask-PS-CTD,Mask-PS-Private_c --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 20260725
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/causal_validation/<model_alias>/<subset>/<task_type>/<intervention>/outputs.json
../cross_task_tool_neurons_data/precise_shield/causal_validation/<model_alias>/<subset>/<task_type>/<intervention>/summary.json
../cross_task_tool_neurons_data/precise_shield/causal_validation/<model_alias>/<subset>/summary_table.csv
../cross_task_tool_neurons_data/precise_shield/causal_validation/<model_alias>/<subset>/cross_type_summary.csv
```

做法：只用 test split 和 base 模型。mask 位置是 `down_proj` 输入 `h`，对选中 intermediate 坐标置零，并作用于所有 token 位置。

## PS-11：结果汇总

单模型：

```text
python PreciseShield/ps_build_final_report.py --model-alias qwen3-4b-instruct --labels-dir ../cross_task_tool_neurons_data/labels --neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --checkpoints-dir ../cross_task_tool_neurons_data/precise_shield/checkpoints --outputs-dir ../cross_task_tool_neurons_data/precise_shield/outputs --causal-dir ../cross_task_tool_neurons_data/precise_shield/causal_validation
```

六模型汇总：

```text
python PreciseShield/ps_build_final_report.py --model-alias all --labels-dir ../cross_task_tool_neurons_data/labels --neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --checkpoints-dir ../cross_task_tool_neurons_data/precise_shield/checkpoints --outputs-dir ../cross_task_tool_neurons_data/precise_shield/outputs --causal-dir ../cross_task_tool_neurons_data/precise_shield/causal_validation
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/outputs/final_report/<model_alias_or_all_models>/README_results.md
../cross_task_tool_neurons_data/precise_shield/outputs/final_report/<model_alias_or_all_models>/model_summary.csv
../cross_task_tool_neurons_data/precise_shield/outputs/final_report/<model_alias_or_all_models>/neuron_discovery_summary.csv
../cross_task_tool_neurons_data/precise_shield/outputs/final_report/<model_alias_or_all_models>/training_comparison.csv
../cross_task_tool_neurons_data/precise_shield/outputs/final_report/<model_alias_or_all_models>/causal_cross_type_summary.csv
../cross_task_tool_neurons_data/precise_shield/outputs/final_report/<model_alias_or_all_models>/figures/*.png
```

做法：只读已有产物，汇总六模型的 PS-TDN/PS-CTD 数量、训练情况、Base/训练后评测、delta 和因果验证结果。

## 六模型运行

每个模型依次从总 README 阶段 2、阶段 3 跑到这里的 PS-4 到 PS-11。命令中只替换：

```text
--model-alias qwen3-4b-instruct
```

替换为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

全部模型完成后运行 PS-11 的 `--model-alias all`。
