# ToolKnowledgeNeurons

TKN 复用根目录 `README.md` 的阶段 1-3，不重新生成标签数据。它的神经元定义沿用 Fei Shen 相关大模型工作中使用的 FFN intermediate 坐标：每层 MLP `down_proj` 输入处的 `h = act(gate_proj(x)) * up_proj(x)`，即 `ffn_intermediate` 的 `(layer, index)`。

TKN-5 只读 `train` split activation。每个任务类型 A/B/C 内按 `env_name, difficulty, id` 确定性配对 `tool_necessary=1` 与 `tool_necessary=0` 样本，计算 `delta = a_tool - a_direct`。每层分数为：

```text
paired_shift = |mean(delta)| / sqrt(std(delta)^2 + floor^2)
z_c = zscore(signed_shift * normalized_down_proj_col_norm)
consensus = max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C))
TKN_score = relu(consensus) * sqrt(min(weighted_shift_A,B,C) * mean(weighted_shift_A,B,C))
```

这相当于把 PreciseShield 的大模型 FFN-intermediate 神经元定义、deepfake-code 的平行样本 paired shift、Safety Kernel 的跨任务共享方向一致性合在一起。可选 `--refine-with-linear-probe` 会在 TKN-5 内部用 train split 临时训练一个线性 probe，只把 `|coef|` 作为 DNA 风格的神经元重排权重；后续 PP-1/PP-2 仍完全沿用 ProbePrefill 的二分类 probe 训练方法，不改训练细节。

## 单卡运行命令

TKN-4 抽取 FFN intermediate activation，`train` 和 `test` 都抽取；TKN-5 只用 `train` 发现神经元，`test` 只供 PP-1 构建 probe/test 特征。

```text
python ToolKnowledgeNeurons/tkn_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 2026
```

```text
python ToolKnowledgeNeurons/tkn_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/visualizations --subset all --selection top_ratio --top-ratio 0.10 --min-neurons-per-layer 64 --min-shared-score 0.0 --min-pairs 2 --max-pairs 0 --epsilon 1.0e-4 --floor-ratio 0.05 --heatmap-top-n 300 --device cuda:0
```

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_neurons --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_neurons --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

重复运行会按 manifest 提前跳过。发现旧产物有误时，在对应 TKN-4、TKN-5、PP-1 或 PP-2 命令末尾追加 `--clean`；清理范围限制在本方案输出目录内。

可选的 DNA 风格临时线性 probe 重排命令：

```text
python ToolKnowledgeNeurons/tkn_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/visualizations --subset all --selection top_ratio --top-ratio 0.10 --min-neurons-per-layer 64 --min-shared-score 0.0 --min-pairs 2 --max-pairs 0 --epsilon 1.0e-4 --floor-ratio 0.05 --heatmap-top-n 300 --device cuda:0 --refine-with-linear-probe --refine-keep-ratio 0.50 --refine-reg 10000 --refine-max-iter 2000
```

不再提供 `--top-ratio` 超过 `0.10` 的 TKN 诊断指令，避免正式实验误用过宽的神经元集合。

## 主要输出

```text
../cross_task_tool_neurons_data/tool_knowledge_neurons/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons/<model_alias>/shared_by_subset/<subset>/TKN_CTD_neurons.jsonl
../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/TKN_TDN_neurons.jsonl
../cross_task_tool_neurons_data/tool_knowledge_neurons/visualizations/<model_alias>/shared_by_subset/*.png
../cross_task_tool_neurons_data/probe_prefill/tool_knowledge_neurons/
```
