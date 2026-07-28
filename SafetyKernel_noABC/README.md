# SafetyKernel_noABC 单卡运行说明

本目录实现 Safety Kernel 的 noABC 版本：不再先找 A/B/C 单类神经元再取交集，而是在每个 subset 的 `train` split 上，直接按模型专属 `tool_necessary` 标签做二分类对比，发现一组全任务工具决策神经元。训练和评测不在本目录做，统一交给 `ProbePrefill`。

## 方法思想

神经元定义保持 Safety Kernel 方案的 FFN module output：

```text
n_i = h_m(x)_i,  m in {gate_proj, up_proj, down_proj}
```

SKNA-5 只读取 `train` activation，构造：

```text
D_call   = train 中 tool_necessary=1 的全部样本
D_direct = train 中 tool_necessary=0 的全部样本
delta_i  = mean(D_call)_i - mean(D_direct)_i
SCAR_i   = zscore(delta_i / pooled_std_i) + zscore(delta_i)
```

最终神经元集合为全模型 FFN 坐标上的 `TopK(SCAR)`：

```text
SK_noABC_TDN = TopK_i(SCAR_i)
```

这里不按 A/B/C 分组，不做 A/B/C 交集；`single_hop` 和 `multi_hop` 仍然严格分开。

## 前置产物

先完成仓库根目录 `README.md` 的阶段 1-3，得到模型专属标签和改造后数据集。阶段 4 起使用本目录命令，输出到独立的 `safety_kernel_noabc` 命名空间。

## SKNA-4：抽取 FFN last-token activation

单卡正式命令：

```text
python SafetyKernel_noABC/skna_extract_ffn_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/safety_kernel_noabc/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

输出：

```text
../cross_task_tool_neurons_data/safety_kernel_noabc/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/safety_kernel_noabc/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/safety_kernel_noabc/activations/<model_alias>/<subset>/<split>/summary.json
../cross_task_tool_neurons_data/safety_kernel_noabc/activations/<model_alias>/<subset>/<split>/manifest.json
```

本阶段复用 Safety Kernel 原 FFN activation 抽取逻辑和 When2Tool prompt 构造：`current/no_reasoning/enable_thinking=false`。已完成且 manifest 参数一致会提前跳过。

## SKNA-5：发现 noABC 工具决策神经元

单卡正式命令：

```text
python SafetyKernel_noABC/skna_discover_tool_decision_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/safety_kernel_noabc/activations --neurons-dir ../cross_task_tool_neurons_data/safety_kernel_noabc/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_noabc/visualizations --subset all --top-k 5000 --heatmap-top-n 300 --epsilon 1.0e-8 --min-class-count 2 --device cuda:0
```

输出：

```text
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/SK_noABC_TDN_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/scar_scores.pt
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/module_meta.json
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/layer_counts.csv
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/module_counts.csv
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/class_balance.csv
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/safety_kernel_noabc/neurons/<model_alias>/shared_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/safety_kernel_noabc/visualizations/<model_alias>/shared_by_subset/*.png
```

打印信息包括 compute device、subset 顺序、逐 FFN module SCAR 进度条、每个 subset 的 `SK_noABC_TDN` 数量和 0/1 标签数量。已完成且 manifest 参数一致会提前跳过，不读取大 activation；旧错误产物可在原命令末尾追加：

```text
--clean
```

清理范围只限 `safety_kernel_noabc` 当前输出。

## 交给 ProbePrefill 训练

SKNA-5 完成后，用 `ProbePrefill` 读取 `SK_noABC_TDN` 训练 logistic probe：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_noabc --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_noabc --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 换模型

只替换命令中的 `--model-alias`：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```
