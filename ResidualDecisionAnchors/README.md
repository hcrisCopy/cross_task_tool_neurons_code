# ResidualDecisionAnchors

RDA 复用根目录 `README.md` 的阶段 1-3，不重新生成标签数据。它把 When2Tool 的全层 hidden states 拆成更细的 residual-state 神经元坐标：`(layer, residual_state, index)`。

## 共享神经元流程

1. RDA-4 对齐 When2Tool 设置，抽取 current prompt、no reasoning、最后输入 token 的全层 `outputs.hidden_states`。
2. RDA-5 只读 `train` split，对 A/B/C 三类任务分别计算每个 residual 维度在 `tool_necessary=1` 与 `tool_necessary=0` 之间的 signed effect。
3. 每层内做 z-score，得到 `z_A,z_B,z_C`。
4. 用方向一致性构造共享分数：

```text
score = max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C))
```

正式实验命令每层取 top 10%，得到 `RDA_CTD`。RDA 只改变送入 ProbePrefill 的神经元集合，不改变 PP-1/PP-2 的二分类 probe 训练方法。

## 单卡运行命令

RDA-4 抽取 residual hidden activation：

```text
python ResidualDecisionAnchors/rda_extract_hidden_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/activations --when2tool-repo third_party/when2tool --subset all --split all --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --device-map auto --max-samples 0
```

RDA-5 发现共享神经元：

```text
python ResidualDecisionAnchors/rda_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/activations --neurons-dir ../cross_task_tool_neurons_data/residual_decision_anchors/neurons --visualizations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/visualizations --subset all --top-ratio 0.10 --min-neurons-per-layer 1 --min-class-count 2 --heatmap-top-n 300 --epsilon 1.0e-6 --device cuda:0
```

PP-1 构建 probe 特征：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method residual_decision_anchors --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

PP-2 训练二分类 probe：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method residual_decision_anchors --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 输出

```text
../cross_task_tool_neurons_data/residual_decision_anchors/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/residual_decision_anchors/neurons/<model_alias>/shared_by_subset/<subset>/RDA_CTD_neurons.jsonl
../cross_task_tool_neurons_data/residual_decision_anchors/visualizations/<model_alias>/shared_by_subset/*.png
../cross_task_tool_neurons_data/probe_prefill/residual_decision_anchors/
```

RDA-5 会输出逐层数量图、全局 top score 热力图和逐层 top 1% score 热力图。重复运行会按 manifest 跳过；旧产物错误时在对应命令末尾追加 `--clean`。
