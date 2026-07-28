# SafetyKernel_Union

本方案只替换 Safety Kernel 的阶段 6 集合算子。总 README 的阶段 1-5 产物全部复用：阶段 5 仍然在 train split 上分别得到 A/B/C 的 `TDN_neurons.jsonl`，SCAR 公式、`top_k=5000`、神经元身份 `(layer, module, index)` 都不改。

原 Safety Kernel 阶段 6 是交集：

```text
CTD = TDN_A intersection TDN_B intersection TDN_C
```

SafetyKernel_Union 阶段 6 改为并集：

```text
CTD_Union = TDN_A union TDN_B union TDN_C
```

这样 ProbePrefill 后续读取的是所有被任一任务类型识别出的工具决策神经元，用来检验“更宽的跨类型工具决策子空间”是否比交集核心更适合二分类 probe。

## 阶段 6 单卡正式命令

```text
python SafetyKernel_Union/sku_discover_union_neurons.py --model-alias qwen3-4b-instruct --input-neurons-dir ../cross_task_tool_neurons_data/neurons --output-neurons-dir ../cross_task_tool_neurons_data/safety_kernel_union/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_union/visualizations --subset all --heatmap-top-n 300
```

`--subset all` 会按顺序处理 `single_hop` 和 `multi_hop`，两者输出分开。换模型只改 `--model-alias`，例如 `qwen3-1.7b`、`qwen3-14b`、`llama3.1-8b`。

## 输入

```text
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/TDN_neurons.jsonl
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/scar_scores.pt
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/module_meta.json
```

## 输出

```text
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/CTD_Union_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/pairwise_<AB|AC|BC>_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/triple_intersection_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/exclusive_<A|B|C>_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/safety_kernel_union/visualizations/<model_alias>/shared_by_subset/*.png
../cross_task_tool_neurons_data/safety_kernel_union/visualizations/<model_alias>/shared_by_subset/ctd_union_layer_top1pct_scar_<min|mean>_heatmap_<subset>.png
../cross_task_tool_neurons_data/safety_kernel_union/visualizations/<model_alias>/single_type_by_subset/sku_layer_top1pct_scar_heatmap_<subset>_<A|B|C>.png
```

终端会打印每个 subset 的 `CTD_Union` 数量、三类交集数量、两两重叠数量和 membership 分布。新增的 `sku_layer_top1pct_scar_heatmap` 复用阶段 5 的 Safety Kernel SCAR 分数，按 `layer + gate_proj/up_proj/down_proj` 展示 A/B/C 单类型每层每个 FFN module 前 1% 神经元分数；`ctd_union_layer_top1pct_scar_<min|mean>_heatmap` 展示 CTD-Union 集合内按 `score_min` / `score_mean` 排序后的共享/并集 top1% 分数。产物存在且 manifest 参数一致时会提前跳过；如果旧 `CTD_Union` 产物存在但缺少新图，重跑本命令会直接补图。

## 重跑与清理

发现旧产物错误时，在原命令末尾追加：

```text
--clean
```

清理范围限制在 `../cross_task_tool_neurons_data/safety_kernel_union/` 对应输出目录和本方案可视化文件内，不会删除阶段 1-5 产物。
