# ToolDecisionAnchors

TDA 复用根目录 `README.md` 的阶段 1-3，不重新生成标签数据。它把神经元定义为 Safety Kernel 风格的大模型 FFN output 坐标：`(layer, module, index)`，默认使用 `gate_proj,up_proj`。

## 共享神经元流程

1. TDA-4 抽取 train/test 的 FFN output activation；TDA-5 只读 `train` split 发现神经元。
2. 对 A/B/C 三类任务分别按标签分组，计算每个神经元 `tool_necessary=1` 相对 `tool_necessary=0` 的标准化 signed effect。
3. 在每个 layer/module 内做 z-score，得到 `z_A,z_B,z_C`。
4. 用方向一致性构造共享分数：

```text
score = max(min(z_A,z_B,z_C), min(-z_A,-z_B,-z_C))
```

如果 A/B/C 都是 tool-high 或都 direct-high，分数高；方向不一致时分数低。正式实验命令每个 layer/module 取 top 10%，得到 `TDA_CTD`，后续 PP-1/PP-2 训练方法不变。

## 单卡运行命令

TDA-4 抽取 FFN output activation：

```text
python ToolDecisionAnchors/tda_extract_ffn_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

TDA-5 发现共享神经元：

```text
python ToolDecisionAnchors/tda_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/activations --neurons-dir ../cross_task_tool_neurons_data/tool_decision_anchors/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/visualizations --subset all --modules gate_proj,up_proj --top-ratio 0.10 --min-neurons-per-module 1 --min-class-count 2 --heatmap-top-n 300 --epsilon 1.0e-6 --device cuda:0
```

PP-1 构建 probe 特征：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_decision_anchors --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

PP-2 训练二分类 probe：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_decision_anchors --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 输出

```text
../cross_task_tool_neurons_data/tool_decision_anchors/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/tool_decision_anchors/neurons/<model_alias>/shared_by_subset/<subset>/TDA_CTD_neurons.jsonl
../cross_task_tool_neurons_data/tool_decision_anchors/visualizations/<model_alias>/shared_by_subset/*.png
../cross_task_tool_neurons_data/probe_prefill/tool_decision_anchors/
```

TDA-5 会输出 density、全局 top score、逐层 mean score，以及逐层 top 1% score 热力图。重复运行会按 manifest 跳过；旧产物错误时在对应命令末尾追加 `--clean`。
