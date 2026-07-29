# ToolRoutingNeurons

ToolRoutingNeurons (TRN) 只替换“探测哪些神经元”这一段，不改变 ProbePrefill 后续的二分类 logistic probe 训练与评测方式。

## 思想

When2Tool 直接拼接全层 residual hidden states，维度大但不区分模型内部功能位置。TRN 采用 Fei Shen 大模型安全迁移工作中使用过的 attention projection neuron 定义：`q_proj/k_proj/v_proj` 的输出坐标，以及 `o_proj` 的输入坐标，即 `(layer, attn_q|attn_k|attn_v|attn_o_in, index)`。这些坐标更贴近“路由哪些信息进入注意力读写”的机制，适合工具调用决策这种需要在问题表征和工具可用性之间做选择的任务。

TRN-4 对齐 When2Tool 的抽取设置：current prompt、no reasoning、`enable_thinking=False`、最后输入 token。TRN-5 只使用 `train` split 发现共享神经元，`test` activation 只供 PP-1 构建 probe/test 特征。

## 分数公式

对每个 task type `c in {A,B,C}`，在 `train` split 内确定性配对 `tool_necessary=1` 和 `tool_necessary=0` 样本：

```text
Delta_i^c = a_i(tool) - a_i(direct)
floor = max(eps, median(std(Delta^c)) * floor_ratio)
signed_i^c = mean(Delta_i^c) / sqrt(std(Delta_i^c)^2 + floor^2)
support_i^c = mean(abs(Delta_i^c) > floor)
w_i = projection_norm_i / mean(projection_norm)
r_i^c = abs(signed_i^c) * w_i * support_i^c^support_power * abs(z_i^c)
z_i^c = zscore(signed_i^c * w_i * support_i^c^support_power)
```

每个 task type 在每层每个 attention routing module 内先取 `r_i^c` 的 top 10%，然后取严格 `A/B/C` 交集。最终共享分数只保留方向一致的神经元：

```text
consensus_z_i = max(min(z_i^A,z_i^B,z_i^C), min(-z_i^A,-z_i^B,-z_i^C))
TRN_score_i = relu(consensus_z_i) * sqrt(min(r_i^A,r_i^B,r_i^C) * mean(r_i^A,r_i^B,r_i^C))
```

这个设计吸收 Safety Kernel 的方向一致共享核心、PreciseShield 的投影权重重要性、deepfake-code/paired data 的配对差分，以及 Fei Shen 系列工作中“稀疏潜在知识坐标”的思路；全程只抽激活和权重范数，不额外训练探测器。

## 输出

```text
../cross_task_tool_neurons_data/tool_routing_neurons/activations/
../cross_task_tool_neurons_data/tool_routing_neurons/neurons/
../cross_task_tool_neurons_data/tool_routing_neurons/visualizations/
../cross_task_tool_neurons_data/probe_prefill/tool_routing_neurons/
```

核心文件：

```text
TRN_CTD_neurons.jsonl
TRN_TDN_neurons.jsonl
trn_scores.pt
top_neurons.csv
layer_summary.csv
module_counts.csv
```

manifest 参数一致时会提前跳过。若需要清理本方案错误旧产物，在对应原命令末尾追加 `--clean`。

## 单卡正式命令

TRN-4：抽取 attention routing activations。

```text
python ToolRoutingNeurons/trn_extract_attention_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_routing_neurons/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --device-map auto --max-samples 0
```

TRN-5：只用 train split 发现 `TRN_CTD`。

```text
python ToolRoutingNeurons/trn_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_routing_neurons/activations --neurons-dir ../cross_task_tool_neurons_data/tool_routing_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_routing_neurons/visualizations --subset all --modules attn_q,attn_k,attn_v,attn_o_in --type-top-ratio 0.10 --min-neurons-per-module 1 --min-shared-score 0.0 --min-consensus-z 0.0 --min-pairs 2 --max-pairs 0 --epsilon 1.0e-4 --floor-ratio 0.05 --support-power 0.5 --heatmap-top-n 300 --device cuda:0
```

PP-1：构建 probe 特征。

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_routing_neurons --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

PP-2：训练 logistic probe。

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_routing_neurons --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```
