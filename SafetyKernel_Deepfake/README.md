# SafetyKernel_Deepfake 单卡运行说明

本目录实现一个新的 Safety Kernel 变体：神经元定义仍使用 Safety Kernel 的 FFN module output 坐标 `(layer, module, index)`，但单类型神经元分数改用 `deepfake-code` 的 paired shift score。训练和评测不在本目录执行，统一交给 `ProbePrefill`。

## 方法思想

Deepfake 原方法对真伪配对样本先计算每个神经元的 paired delta，再按通道统计：

```text
delta_i(pair) = activation_fake_i - activation_real_i
score_i = abs(mean(delta_i)) / sqrt(std(delta_i)^2 + floor^2)
floor = max(epsilon, median(valid std_delta) * floor_ratio)
```

迁移到工具调用任务时，本方案只改分数计算，不改神经元空间：

```text
n_i = h_m(x)_i,  m in {gate_proj, up_proj, down_proj}
delta_i(pair) = h_m(tool_necessary=1)_i - h_m(tool_necessary=0)_i
```

每个 subset 内只读取 `train` activation，A/B/C 三类任务分别构造确定性一对一 label pair：先在同一 task type 内按 `env_name, difficulty, id` 排序，再取 `min(n_label1, n_label0)` 对。随后每个 FFN module 按 Deepfake 的 per-layer/per-module `top_ratio` 规则选神经元，得到 `SKD_TDN_A/B/C`，最后按完整坐标取交集：

```text
SKD_CTD = SKD_TDN_A intersection SKD_TDN_B intersection SKD_TDN_C
```

`single_hop` 和 `multi_hop` 始终分开处理。阶段 1-3 复用根目录 `README.md` 的模型专属标签和改造后数据集产物。

## SKD-4：抽取 FFN last-token activation

单卡正式命令：

```text
python SafetyKernel_Deepfake/skdf_extract_ffn_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

输出：

```text
../cross_task_tool_neurons_data/safety_kernel_deepfake/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/safety_kernel_deepfake/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/safety_kernel_deepfake/activations/<model_alias>/<subset>/<split>/summary.json
../cross_task_tool_neurons_data/safety_kernel_deepfake/activations/<model_alias>/<subset>/<split>/manifest.json
```

本阶段复用根方案的 Safety Kernel FFN output 抽取逻辑和 When2Tool prompt 构造：`current/no_reasoning/enable_thinking=false`。产物存在且 manifest 参数一致会提前跳过。

## SKD-5：A/B/C 单类型 paired-shift 神经元发现

单卡正式命令：

```text
python SafetyKernel_Deepfake/skdf_discover_single_type_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/activations --neurons-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations --subset all --top-ratio 0.01 --min-neurons-per-module 1 --heatmap-top-n 300 --epsilon 1.0e-4 --floor-ratio 0.05 --min-pairs 2 --max-pairs 0 --device cuda:0
```

输出：

```text
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/SKD_TDN_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/deepfake_scores.pt
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/pair_meta.jsonl
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/layer_summary.csv
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/top_neurons.csv
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/summary.json
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/module_meta.json
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/class_balance.csv
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations/<model_alias>/single_type_by_subset/*.png
```

终端会打印 compute device、subset 顺序、逐 module 进度条、每个 task type 的 `SKD_TDN` 数量和 pair 数量。可视化对齐 Deepfake：score heatmap 色条从 0 开始，并保存 layer shift 曲线和 selected count 图。

## SKD-6：A/B/C 交集共享神经元

单卡正式命令：

```text
python SafetyKernel_Deepfake/skdf_discover_shared_neurons.py --model-alias qwen3-4b-instruct --neurons-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations --subset all --heatmap-top-n 300
```

输出：

```text
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/SKD_CTD_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/pairwise_<AB|AC|BC>_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/layer_counts.csv
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/module_counts.csv
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/share_rates.csv
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations/<model_alias>/shared_by_subset/*.png
```

共享强度使用保守分数：

```text
shared_score = score_min = min(score_A, score_B, score_C)
```

终端会打印每个 subset 的 `SKD_CTD` 数量和两两重叠数量。产物存在且 manifest 参数一致会提前跳过。

## 交给 ProbePrefill

完成 SKD-6 后，使用 `safety_kernel_deepfake` 方法名构建 probe 特征并训练：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_deepfake --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_deepfake --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 重跑与清理

正常重复运行会按 manifest 提前跳过。发现旧产物错误时，在对应原命令末尾追加：

```text
--clean
```

清理范围限制在 `../cross_task_tool_neurons_data/safety_kernel_deepfake/` 当前阶段输出和本方案可视化文件内，不会删除阶段 1-3 产物。

## 换模型

只改 `--model-alias`：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```
