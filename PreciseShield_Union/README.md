# PreciseShield_Union

本目录只替换 PreciseShield 的阶段 6 集合算子。PreciseShield 阶段 4/5 的激活抽取、PS saliency 公式、`PS_TDN_c = TopK(S_call,c) \ TopK(S_direct,c)` 产物全部复用，不重新抽 activation，不修改神经元定义。

原 PreciseShield 阶段 6 使用交集：

```text
PS_CTD = PS_TDN_A intersection PS_TDN_B intersection PS_TDN_C
```

PreciseShield_Union 阶段 6 改为并集：

```text
PS_CTD_Union = PS_TDN_A union PS_TDN_B union PS_TDN_C
```

这样 ProbePrefill 后续读取的是所有被任一任务类型识别出的 FFN intermediate 工具决策神经元，用来检验更宽的跨类型工具决策子空间是否更适合二分类 probe。

## 阶段 6 单卡正式命令

```text
python PreciseShield_Union/psu_discover_union_neurons.py --model-alias qwen3-4b-instruct --input-neurons-dir ../cross_task_tool_neurons_data/precise_shield/neurons --output-neurons-dir ../cross_task_tool_neurons_data/precise_shield_union/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield_union/visualizations --subset all --heatmap-top-n 300
```

`--subset all` 会按顺序处理 `single_hop` 和 `multi_hop`，两者输出分开保存。换模型只改 `--model-alias`，可替换为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

## 输入

```text
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/PS_TDN_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/saliency_scores.pt
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/module_meta.json
```

阶段 6 只使用 train split 产生的阶段 5 PS-TDN，不读取 test，不重新 split。

## 输出

```text
../cross_task_tool_neurons_data/precise_shield_union/neurons/<model_alias>/shared_by_subset/<subset>/PS_CTD_Union_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_union/neurons/<model_alias>/shared_by_subset/<subset>/pairwise_<AB|AC|BC>_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_union/neurons/<model_alias>/shared_by_subset/<subset>/triple_intersection_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_union/neurons/<model_alias>/shared_by_subset/<subset>/exclusive_<A|B|C>_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_union/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/precise_shield_union/neurons/<model_alias>/shared_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/precise_shield_union/visualizations/<model_alias>/shared_by_subset/*.png
../cross_task_tool_neurons_data/precise_shield_union/visualizations/<model_alias>/single_type_by_subset/psu_layer_top1pct_saliency_heatmap_<subset>_<A|B|C>.png
```

终端会打印每个 subset 的 `PS_CTD_Union` 数量、三类交集数量、两两重叠数量和 membership 分布。产物存在且 manifest 参数一致时提前跳过；如旧产物错误，在原命令末尾追加：

```text
--clean
```

清理范围限制在 `../cross_task_tool_neurons_data/precise_shield_union/` 对应输出内，不会删除 PreciseShield 阶段 4/5 产物。
