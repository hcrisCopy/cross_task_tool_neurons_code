# ResidualDecisionAnchors

ResidualDecisionAnchors (RDA) is a residual-stream neuron variant for tool-call decisions. It keeps the ProbePrefill logistic probe unchanged, but replaces the coarse When2Tool "concatenate all hidden states" feature with selected `(layer, residual_state, index)` dimensions.

## Score

For each residual layer and dimension:

```text
effect_c(i) = (mean_tool(i) - mean_direct(i)) / sqrt(var_tool(i) + var_direct(i) + epsilon)
z_c(i) = zscore(effect_c(i)) within the same residual layer
score(i) = max(min(z_A(i), z_B(i), z_C(i)), min(-z_A(i), -z_B(i), -z_C(i)))
```

RDA-CTD keeps the per-layer Top-Ratio dimensions by this signed A/B/C consensus score.

## Commands

Reuse root `README.md` stages 1-3 first.

RDA-4:

```text
python ResidualDecisionAnchors/rda_extract_hidden_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/activations --when2tool-repo third_party/when2tool --subset all --split all --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --device-map auto --max-samples 0
```

RDA-5:

```text
python ResidualDecisionAnchors/rda_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/activations --neurons-dir ../cross_task_tool_neurons_data/residual_decision_anchors/neurons --visualizations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/visualizations --subset all --top-ratio 0.80 --min-neurons-per-layer 1 --min-class-count 2 --heatmap-top-n 300 --epsilon 1.0e-6 --device cuda:0
```

PP-1:

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method residual_decision_anchors --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

PP-2:

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method residual_decision_anchors --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```
