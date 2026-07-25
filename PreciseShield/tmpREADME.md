# PreciseShield 小样本跑通记录

本文件只记录 `PreciseShield/` 新方案。旧方案代码不动，阶段 1-3 仍然用总 README 的产物。下面命令默认在仓库根目录运行。

本次先打通：

```text
model = qwen3-4b-instruct
single_hop train/test = 100 / 30
multi_hop train/test  = 40 / 30
sample_strategy = balanced
prompt = When2Tool current/no_reasoning
enable_thinking = false
tool format = 自动按 When2Tool 规则识别，Qwen 为 XML，Llama 为 native
```

所有 PreciseShield 新产物默认写到：

```text
../cross_task_tool_neurons_data/precise_shield/
```

重复运行时，如果 manifest 参数一致会提前跳过。发现旧产物错了，在对应命令末尾加 `--clean` 清理当前阶段目标产物后重跑；想强制覆盖但不先清目录，加 `--overwrite`。

## 0. 前置阶段 1-3

这些不是 PreciseShield 新代码，只是新方案依赖它们的产物。

涉及文件：

```text
code/01_raw_data_preparation/inspect_raw_data.py
code/02_labeling/generate_tool_necessity_labels.py
code/03_dataset_building/build_modified_dataset.py
third_party/when2tool/
```

运行：

```text
python code/01_raw_data_preparation/inspect_raw_data.py --overwrite
```

```text
python code/02_labeling/generate_tool_necessity_labels.py --model-alias qwen3-4b-instruct --raw-dataset-dir ../cross_task_tool_neurons_data/datasets/raw_when2tool --labels-dir ../cross_task_tool_neurons_data/labels --when2tool-repo third_party/when2tool --single-train-count 100 --single-test-count 30 --multi-train-count 40 --multi-test-count 30 --candidate-multiplier 2.0 --require-per-type-labels --backend vllm --tensor-parallel-size 1 --max-model-len 32768 --max-new-tokens 2048 --max-rounds 12 --record-mode lite --vllm-dtype bfloat16 --seed 2026
```

```text
python code/03_dataset_building/build_modified_dataset.py --model-alias qwen3-4b-instruct --raw-dataset-dir ../cross_task_tool_neurons_data/datasets/raw_when2tool --labels-dir ../cross_task_tool_neurons_data/labels --output-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool
```

输出：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/single_hop/train.jsonl
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/single_hop/test.jsonl
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/multi_hop/train.jsonl
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/multi_hop/test.jsonl
```

方法说明：阶段 2 生成二分类标签 `tool_necessary`，阶段 3 把原始题目、tool schema、env 信息和标签合并。PreciseShield 从这里往后接。

## PS-4. 抽取 FFN intermediate 激活

涉及文件：

```text
PreciseShield/ps_common.py
PreciseShield/ps_extract_intermediate_activations.py
third_party/when2tool/
```

运行：

```text
python PreciseShield/ps_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset single_hop --split train --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 100 --sample-strategy balanced --seed 2026 --device-map auto
```

```text
python PreciseShield/ps_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset single_hop --split test --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 30 --sample-strategy balanced --seed 2026 --device-map auto
```

```text
python PreciseShield/ps_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset multi_hop --split train --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 40 --sample-strategy balanced --seed 2026 --device-map auto
```

```text
python PreciseShield/ps_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset multi_hop --split test --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 30 --sample-strategy balanced --seed 2026 --device-map auto
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/activations/qwen3-4b-instruct/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/precise_shield/activations/qwen3-4b-instruct/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/precise_shield/activations/qwen3-4b-instruct/<subset>/<split>/summary.json
../cross_task_tool_neurons_data/precise_shield/activations/qwen3-4b-instruct/<subset>/<split>/manifest.json
```

方法说明：用 When2Tool `init_state` 构造 prompt，hook 每层 `down_proj` 的输入，保存最后一个输入 token 的 FFN intermediate `h`。`down_weight_norms` 也一起保存，供 PreciseShield 分数计算使用。阶段 5 只用 train 激活；test 激活先保存下来，方便检查和后续扩展。

## PS-5. A/B/C 单类型神经元探测

涉及文件：

```text
PreciseShield/ps_discover_single_type_neurons.py
PreciseShield/ps_common.py
```

运行：

```text
python PreciseShield/ps_discover_single_type_neurons.py --model-alias qwen3-4b-instruct --subset all --intervention-ratio 0.05 --heatmap-top-n 300 --epsilon 1.0e-12 --min-class-count 2
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/single_type_by_subset/<subset>/<A|B|C>/PS_TDN_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/single_type_by_subset/<subset>/<A|B|C>/saliency_scores.pt
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/single_type_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/precise_shield/visualizations/qwen3-4b-instruct/single_type_by_subset/*.png
```

方法说明：对每个类型分别计算 `D_call` 和 `D_direct` 的 PreciseShield saliency：

```text
I_i(D) = |mean(h_i(D))| * ||W_down[:, i]||_2
S_i(D) = I_i(D) / (sum_j I_j(D) + epsilon)
PS_TDN_c = TopK(S_call,c) \ TopK(S_direct,c)
```

这里小样本用 `--intervention-ratio 0.05`，是为了先保证交集不太容易为空。正式实验建议回到 `0.01`。

## PS-6. 跨类型共享神经元

涉及文件：

```text
PreciseShield/ps_discover_shared_neurons.py
PreciseShield/ps_common.py
```

运行：

```text
python PreciseShield/ps_discover_shared_neurons.py --model-alias qwen3-4b-instruct --subset all --heatmap-top-n 300
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/PS_CTD_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/pairwise_AB_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/pairwise_AC_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/pairwise_BC_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/private_A_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/private_B_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/qwen3-4b-instruct/shared_by_subset/<subset>/private_C_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/visualizations/qwen3-4b-instruct/shared_by_subset/*.png
```

方法说明：

```text
PS_CTD = PS_TDN_A intersection PS_TDN_B intersection PS_TDN_C
```

训练只使用 `PS_CTD_neurons.jsonl`。pairwise 和 private 文件只用于分析和因果对照。

## PS-7. PreciseShield-Masked-LoRA 训练

涉及文件：

```text
PreciseShield/ps_train_masked_lora.py
PreciseShield/ps_common.py
third_party/when2tool/
```

单跳训练：

```text
python PreciseShield/ps_train_masked_lora.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset single_hop --max-train-samples 100 --sample-strategy balanced --rank 8 --lora-alpha 16 --lora-dropout 0 --epochs 1 --per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 5e-5 --warmup-ratio 0.03 --max-grad-norm 1.0 --max-seq-length 4096 --trajectory-attempts 2 --trajectory-batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode full
```

多跳训练：

```text
python PreciseShield/ps_train_masked_lora.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset multi_hop --max-train-samples 40 --sample-strategy balanced --rank 8 --lora-alpha 16 --lora-dropout 0 --epochs 1 --per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 5e-5 --warmup-ratio 0.03 --max-grad-norm 1.0 --max-seq-length 4096 --trajectory-attempts 2 --trajectory-batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode full
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/checkpoints/qwen3-4b-instruct/ps_masked_lora/<subset>/adapter/adapter_model.pt
../cross_task_tool_neurons_data/precise_shield/checkpoints/qwen3-4b-instruct/ps_masked_lora/<subset>/adapter/adapter_config.json
../cross_task_tool_neurons_data/precise_shield/checkpoints/qwen3-4b-instruct/ps_masked_lora/<subset>/training_examples.jsonl
../cross_task_tool_neurons_data/precise_shield/checkpoints/qwen3-4b-instruct/ps_masked_lora/<subset>/skipped_examples.jsonl
../cross_task_tool_neurons_data/precise_shield/checkpoints/qwen3-4b-instruct/ps_masked_lora/<subset>/training_log.csv
../cross_task_tool_neurons_data/precise_shield/checkpoints/qwen3-4b-instruct/ps_masked_lora/<subset>/manifest.json
```

方法说明：只在 FFN `gate_proj/up_proj` 的 PS-CTD 对应行上训练 LoRA：

```text
W' = W0 + (M * B) A
```

`tool_necessary=0` 的 target 是直接 boxed answer；`tool_necessary=1` 的 target 是当前模型第一步成功 tool call 的原始输出。loss 只算 assistant target tokens。

## PS-8. 训练后评测

涉及文件：

```text
PreciseShield/ps_evaluate_trained_model.py
PreciseShield/ps_common.py
third_party/when2tool/
```

运行：

```text
python PreciseShield/ps_evaluate_trained_model.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset all --max-test-samples 30 --sample-strategy balanced --n-runs 1 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/outputs.json
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/per_task.jsonl
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/summary.json
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/summary_table.csv
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/manifest.json
```

方法说明：只用 test split，加载 PS-7 对应 subset 的 adapter，按 When2Tool 官方流程跑工具调用和最终答案评测。

## PS-9. Base 评测与 delta

涉及文件：

```text
PreciseShield/ps_evaluate_base_and_delta.py
PreciseShield/ps_common.py
third_party/when2tool/
```

运行：

```text
python PreciseShield/ps_evaluate_base_and_delta.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset all --max-test-samples 30 --sample-strategy balanced --n-runs 1 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/base_evaluation/<subset>/summary.json
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/base_evaluation/<subset>/summary_table.csv
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/comparison_with_base.csv
../cross_task_tool_neurons_data/precise_shield/outputs/qwen3-4b-instruct/trained_evaluation/<subset>/comparison_with_base_manifest.json
```

方法说明：Base 不加载 adapter。delta 使用和 PS-8 同一批 test rows，重点看 `decision_accuracy`、`tool_call_rate`、`over_call_rate`、`under_call_rate` 和最终答案准确率。

## PS-10. 因果验证

涉及文件：

```text
PreciseShield/ps_causal_validation.py
PreciseShield/ps_common.py
third_party/when2tool/
```

运行：

```text
python PreciseShield/ps_causal_validation.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --subset all --max-test-samples 30 --sample-strategy balanced --interventions Base,Mask-Random,Mask-PS-TDN_c,Mask-PS-CTD,Mask-PS-Private_c --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 2026
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/causal_validation/qwen3-4b-instruct/<subset>/<task_type>/<intervention>/outputs.json
../cross_task_tool_neurons_data/precise_shield/causal_validation/qwen3-4b-instruct/<subset>/<task_type>/<intervention>/per_task.jsonl
../cross_task_tool_neurons_data/precise_shield/causal_validation/qwen3-4b-instruct/<subset>/<task_type>/<intervention>/summary.json
../cross_task_tool_neurons_data/precise_shield/causal_validation/qwen3-4b-instruct/<subset>/summary_table.csv
../cross_task_tool_neurons_data/precise_shield/causal_validation/qwen3-4b-instruct/<subset>/cross_type_summary.csv
```

方法说明：在 `down_proj` 输入处把选中 FFN intermediate 坐标置零。`Mask-Random` 使用和 PS-CTD 相同的每层数量；`Mask-PS-TDN_c`、`Mask-PS-Private_c` 按当前 task type 选择对应集合。

## 11. 换模型

六个模型都用同一套脚本，只替换：

```text
--model-alias qwen3-4b-instruct
```

可替换为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

注意：不同模型的标签、改造后数据集、激活、神经元、adapter、评测输出都按 `model_alias` 隔离，不能混用。
