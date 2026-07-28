# ToolDecisionAnchors

ToolDecisionAnchors (TDA) is a lightweight shared-neuron discovery scheme for tool-call decisions. It keeps the downstream ProbePrefill logistic probe unchanged and only changes which LLM-internal FFN output neurons are selected before PP-1.

## Idea

TDA follows the large-model neuron coordinate used by Safety Kernel: each neuron is an FFN output element identified by `(layer, module, index)` over `gate_proj`, `up_proj`, and optionally `down_proj`. The score borrows the parallel-pair idea used in Safety Kernel, PreciseShield, deepfake-style paired shifts, and Fei Shen/NUS latent sparse-neuron work: each task type should expose a consistent latent tool-decision direction, and the final shared set should require A/B/C agreement instead of a loose union.

For each task type `c in {A,B,C}` and each module:

```text
effect_c(i) = (mean_tool(i) - mean_direct(i)) / sqrt(var_tool(i) + var_direct(i) + epsilon)
z_c(i) = zscore(effect_c(i)) within the same layer/module
score(i) = max(min(z_A(i), z_B(i), z_C(i)), min(-z_A(i), -z_B(i), -z_C(i)))
```

The first `min` keeps neurons where tool-needed prompts are consistently higher across A/B/C. The second keeps neurons where direct-answer prompts are consistently higher across A/B/C. The selected TDA-CTD neurons are the per-module Top-Ratio neurons by this signed consensus score.

## Reused Inputs

Run root `README.md` stages 1-3 first. TDA never rebuilds labels or resplits data. TDA-5 discovers neurons from `train` activations only; PP-1 later reads both `train` and `test` activations to build probe features.

## TDA-4 Extract FFN Activations

```text
python ToolDecisionAnchors/tda_extract_ffn_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

Output:

```text
../cross_task_tool_neurons_data/tool_decision_anchors/activations/<model_alias>/<single_hop|multi_hop>/<train|test>/
```

## TDA-5 Discover Shared Neurons

```text
python ToolDecisionAnchors/tda_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/activations --neurons-dir ../cross_task_tool_neurons_data/tool_decision_anchors/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/visualizations --subset all --modules gate_proj,up_proj --top-ratio 0.70 --min-neurons-per-module 1 --min-class-count 2 --heatmap-top-n 300 --epsilon 1.0e-6 --device cuda:0
```

Main output consumed by ProbePrefill:

```text
../cross_task_tool_neurons_data/tool_decision_anchors/neurons/<model_alias>/shared_by_subset/<single_hop|multi_hop>/TDA_CTD_neurons.jsonl
```

Diagnostic outputs:

```text
../cross_task_tool_neurons_data/tool_decision_anchors/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/tool_decision_anchors/neurons/<model_alias>/shared_by_subset/<subset>/top_neurons.csv
../cross_task_tool_neurons_data/tool_decision_anchors/visualizations/<model_alias>/shared_by_subset/
```

Use `--clean` on the same command to remove stale TDA outputs before recomputing. Normal reruns skip early when the manifest matches.

## PP-1 Build Probe Features

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_decision_anchors --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

## PP-2 Train Logistic Probe

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_decision_anchors --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

PP-2 keeps the When2Tool-style L2 logistic probe unchanged. Single-hop and multi-hop probes are trained and reported separately.

## Model Switching

Only change `--model-alias`, and pass `--model-path` only when the alias and checkpoint really match. For Qwen3 models, keep the model path on Qwen3 checkpoints and do not route them through old Qwen2.5 paths.
