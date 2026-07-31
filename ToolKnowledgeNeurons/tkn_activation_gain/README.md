# TKNActivationGain

本方案复用 `ToolKnowledgeNeurons` 已有 TKN-4/TKN-5 产物，不重新跑模型，不覆盖原 TKN 输出。

核心思想：全局把 activation 乘 `(1 + lambda)` 会被 `StandardScaler + LogisticRegression` 基本抵消，所以这里改成逐神经元、逐样本的方向对齐证据特征。默认 `dual_evidence` 会为每个 TKN 神经元写三列：

```text
signed_z = (sign * activation - train_midpoint) / train_scale
positive_gain = lambda * relu(signed_z) ** power
negative_gain = lambda * relu(-signed_z) ** power
```

`train` split 只用于冻结每个神经元的方向、阈值和尺度；`test` split 只用这些冻结统计量构建特征。PP-2 仍然使用原来的 `StandardScaler + L2 LogisticRegression`，训练细节不变。

## 前置产物

先完成根目录阶段 1-3，以及 `ToolKnowledgeNeurons` 的 TKN-4/TKN-5：

```text
../cross_task_tool_neurons_data/tool_knowledge_neurons/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons/<model_alias>/shared_by_subset/<subset>/TKN_CTD_neurons.jsonl
```

## 单卡正式命令

构建 TKAG 神经元说明、可视化和 ProbePrefill 特征：

```text
python ToolKnowledgeNeurons/tkn_activation_gain/tkag_build_probe_features.py --model-alias qwen3-4b-instruct --subset all --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --tkn-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --output-neurons-dir ../cross_task_tool_neurons_data/tkn_activation_gain/neurons --output-probe-root ../cross_task_tool_neurons_data/probe_prefill/tkn_activation_gain --visualizations-dir ../cross_task_tool_neurons_data/tkn_activation_gain/visualizations --keep-ratio 1.0 --gain-lambda 1.0 --evidence-power 2.0 --feature-mode dual_evidence --threshold-mode midpoint --min-neurons-per-layer 1 --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026 --device cuda:0
```

训练二分类探针：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tkn_activation_gain --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

常用可调参数：

```text
--feature-mode dual_evidence
--gain-lambda 1.0
--evidence-power 2.0
--keep-ratio 1.0
--append-layer-pool
```

`feature-mode` 也支持 `signed_z_gain`、`augmented_gain`、`gaussian_llr`、`augmented_llr`。如果更重视 single-hop AUROC，可以尝试 `--feature-mode augmented_gain --gain-lambda 3.0`。

重复运行时，manifest 一致会提前跳过。发现错误旧产物时，在构建命令末尾追加 `--clean`，只会清理 `tkn_activation_gain` 本方案输出目录和对应 ProbePrefill 子目录，不会删除原 TKN 产物。

## 输出

```text
../cross_task_tool_neurons_data/tkn_activation_gain/neurons/<model_alias>/shared_by_subset/<subset>/TKAG_CTD_neurons.jsonl
../cross_task_tool_neurons_data/tkn_activation_gain/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/tkn_activation_gain/visualizations/<model_alias>/shared_by_subset/*.png
../cross_task_tool_neurons_data/probe_prefill/tkn_activation_gain/probe_features/<model_alias>/<subset>/train_features.pt
../cross_task_tool_neurons_data/probe_prefill/tkn_activation_gain/probe_features/<model_alias>/<subset>/test_features.pt
../cross_task_tool_neurons_data/probe_prefill/tkn_activation_gain/probes/<model_alias>/<subset>/probe_results_no_reasoning.json
```
