# ToolKnowledgePathways: TKN 路径增强方案

本方案只读 `ToolKnowledgeNeurons` 已有产物，不改 TKN、不改旧路径代码。输出写到独立目录：

```text
../cross_task_tool_neurons_data/tool_knowledge_pathways/
../cross_task_tool_neurons_data/probe_prefill/tool_knowledge_pathways/
```

## 思想

论文里的路径分析不是把单个神经元孤立使用，而是先定位任务相关神经元，再用两类边描述跨层协同：

- Co-Activation：目标行为样本里两个神经元共同激活，非目标样本里的通用共激活要扣掉。
- Activation-Propagation：mask 前层源神经元后，后层目标神经元的方向性激活发生下降，说明这条边有因果传播关系。

迁移到工具调用任务时，路径要同时建两条方向：

- `tool_high`：`tool_necessary=1` 的用工具路径。
- `direct_high`：`tool_necessary=0` 的直接回答路径。

旧方案只保留 `tool_high` 且只连相邻层，容易在 25-30 层断开，也会丢掉大量高分 TKN 神经元。新方案改为：

- 每层每方向只取高分 TKN 候选，控制路径搜索开销。
- 允许跨 1-4 层跳连，避免相邻层某一层断边导致后层完全消失。
- 只对分层短名单边做 masking 因果验证，不对所有候选两两验证。
- 最终保留“高分 TKN 锚点 + 路径节点”，路径用于增强和重排，不把 TKN 粗暴过滤成很小的图。

发现阶段只用 `train` split；`test` 只在 ProbePrefill 构建测试特征和评测时使用。

## 单卡正式命令

先确保 TKN-4/TKN-5 已完成。如果还没有，按原 TKN 命令先跑：

```text
python ToolKnowledgeNeurons/tkn_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 2026
```

```text
python ToolKnowledgeNeurons/tkn_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/visualizations --subset all --selection top_ratio --top-ratio 0.10 --min-neurons-per-layer 64 --min-shared-score 0.0 --min-pairs 2 --max-pairs 0 --epsilon 1.0e-4 --floor-ratio 0.05 --heatmap-top-n 300 --device cuda:0
```

TKP-1：构建 TKN 路径增强神经元，`single_hop` 和 `multi_hop` 会分开输出：

```text
python ToolKnowledgePathways/tkn_pathways/tkp_tkn_discover_pathways.py --model-alias qwen3-4b-instruct --subset all --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --tkn-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --output-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_pathways/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_pathways/visualizations --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --gpus 0 --device cuda:0 --candidate-per-direction-per-layer 256 --anchor-per-direction-per-layer 96 --final-per-direction-per-layer 192 --max-layer-gap 4 --edge-top-k 4 --activation-quantile 0.70 --min-target-phi 0.02 --generic-penalty 0.5 --min-edge-score -0.01 --causal-mode sampled_mask --causal-sources-per-layer-direction 3 --causal-targets-per-source 4 --causal-samples-per-task 12 --causal-batch-size 2 --min-causal-effect 0.0 --torch-dtype bfloat16 --device-map auto
```

PP-1：构建 ProbePrefill 特征：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_pathways --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

PP-2：训练 when2tool 风格二分类 probe：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_pathways --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

重复运行时，如果 manifest 参数一致会提前跳过。发现旧错误产物时，在对应命令末尾加 `--clean`，清理范围只限本方案输出目录。

## 主要输出

```text
../cross_task_tool_neurons_data/tool_knowledge_pathways/neurons/<model_alias>/shared_by_subset/<subset>/TKP_TKN_CTD_neurons.jsonl
../cross_task_tool_neurons_data/tool_knowledge_pathways/neurons/<model_alias>/shared_by_subset/<subset>/TKP_TKN_path_edges.jsonl
../cross_task_tool_neurons_data/tool_knowledge_pathways/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/tool_knowledge_pathways/visualizations/<model_alias>/shared_by_subset/<subset>/*.png
../cross_task_tool_neurons_data/probe_prefill/tool_knowledge_pathways/
```
