# PreciseShield_Deepfake 单卡运行说明

本目录实现一个新的 PreciseShield 变体：神经元空间仍然是 PreciseShield 的 FFN intermediate 坐标 `(layer, ffn_intermediate, index)`，但 A/B/C 单类型神经元的分数改用 `deepfake-code` 的 paired shift score。训练和评测不在本目录执行，统一交给 `ProbePrefill`。

## 方法思想

PreciseShield 的神经元定义：

```text
h = sigma(W_gate x + b_gate) * (W_up x + b_up)
n_i^(l) = h_i^(l)
```

PSDF-4 抓取模型开始生成前最后一个输入 token 的 `h`。PSDF-5 只读取 `train` activation，在每个 subset 内对 A/B/C 三类任务分别构造确定性 label pair：

```text
delta_i(pair) = h_i(tool_necessary=1) - h_i(tool_necessary=0)
score_i = abs(mean(delta_i)) / sqrt(std(delta_i)^2 + floor^2)
floor = max(epsilon, median(valid std_delta) * floor_ratio)
```

每层按 deepfake-code 默认比例取 top 10%：

```text
PSDF_TDN_c^(l) = TopRatio_l(score_c, top_ratio)
PSDF_CTD = PSDF_TDN_A intersection PSDF_TDN_B intersection PSDF_TDN_C
```

`single_hop` 和 `multi_hop` 始终分开处理。阶段 1-3 复用根目录 `README.md` 的模型专属标签和改造后数据集产物。

## PSDF-4：抽取 FFN intermediate activation

单卡正式命令：

```text
python PreciseShield_Deepfake/psdf_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 2026
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield_deepfake/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/precise_shield_deepfake/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/precise_shield_deepfake/activations/<model_alias>/<subset>/<split>/summary.json
../cross_task_tool_neurons_data/precise_shield_deepfake/activations/<model_alias>/<subset>/<split>/manifest.json
```

本阶段复用 PreciseShield 的 hook：在 `down_proj` 的 `forward_pre_hook` 处保存 FFN intermediate `h`，prompt 仍由 When2Tool `init_state` 构造，固定 `current/no_reasoning/enable_thinking=false`。

## PSDF-5：A/B/C paired-shift 神经元发现

单卡正式命令：

```text
python PreciseShield_Deepfake/psdf_discover_single_type_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/activations --neurons-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations --subset all --top-ratio 0.10 --min-neurons-per-layer 1 --heatmap-top-n 300 --epsilon 1.0e-4 --floor-ratio 0.05 --min-pairs 2 --max-pairs 0 --device cuda:0
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/PSDF_TDN_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/deepfake_scores.pt
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/pair_meta.jsonl
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/layer_summary.csv
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/top_neurons.csv
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/module_counts.csv
../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations/<model_alias>/single_type_by_subset/*.png
```

终端会打印 compute device、subset 顺序、每层进度条、每个 task type 的 `PSDF_TDN` 数量和 pair 数量。可视化包含 score heatmap、layer shift 曲线和 selected count 图，heatmap 色条从 0 开始以对齐 deepfake-code。

## PSDF-6：A/B/C 交集共享神经元

单卡正式命令：

```text
python PreciseShield_Deepfake/psdf_discover_shared_neurons.py --model-alias qwen3-4b-instruct --neurons-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations --subset all --heatmap-top-n 300
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/PSDF_CTD_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/pairwise_<AB|AC|BC>_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/layer_counts.csv
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/module_counts.csv
../cross_task_tool_neurons_data/precise_shield_deepfake/neurons/<model_alias>/shared_by_subset/<subset>/share_rates.csv
../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations/<model_alias>/shared_by_subset/*.png
```

共享强度使用保守分数：

```text
shared_score = score_min = min(score_A, score_B, score_C)
```

如果某个 subset 出现 `PSDF_CTD=0`，说明严格三交集在当前 `top_ratio` 下为空。代码不会自动改成并集或两两交集；需要调大 PSDF-5 的 `--top-ratio` 后重跑 PSDF-5/PSDF-6。

## 交给 ProbePrefill

完成 PSDF-6 后，使用 `precise_shield_deepfake` 方法名构建 probe 特征并训练：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield_deepfake --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield_deepfake --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 重跑与清理

正常重复运行会按 `manifest.json` 提前跳过。发现旧产物参数错误时，在对应命令末尾追加：

```text
--clean
```

清理范围限制在 `../cross_task_tool_neurons_data/precise_shield_deepfake/` 当前阶段输出和本方案可视化文件内，不会删除阶段 1-3 产物，也不会影响其他方案目录。

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
