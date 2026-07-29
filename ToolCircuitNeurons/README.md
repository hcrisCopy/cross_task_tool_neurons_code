# ToolCircuitNeurons

ToolCircuitNeurons (TCN) 是一个后置融合方案：不改变 ProbePrefill 的二分类 logistic probe 训练方式，只改变“给 probe 哪些神经元特征”。它复用已经完成的 `ToolDecisionAnchors`、`ToolKnowledgeNeurons`、`ToolRoutingNeurons` 的 PP-1 特征，严格校验 `train/test` 的题目 id 和标签顺序后拼接为 `TCN_CTD`。

## 思想

单独的 attention routing 神经元虽然在 A/B/C 交集上分数明显变高，但训练 probe 仍低于 When2Tool，说明工具调用决策不是只落在一个模块空间里的单点神经元，而更像分布式潜在知识 circuit。

TCN 使用三类 Fei Shen 大模型潜在知识/安全迁移工作中相近的神经元定义：

- `TDA_CTD`：Safety Kernel 风格的 FFN output 坐标 `(layer, gate_proj|up_proj, index)`，更贴近最终写入 residual stream 的工具决策锚点。
- `TKN_CTD`：PreciseShield/latent-neuron 风格的 FFN intermediate `h` 坐标 `(layer, ffn_intermediate, index)`，更贴近 MLP 内部知识表征。
- `TRN_CTD`：Fei Shen 大模型安全迁移工作使用过的 attention projection row/column routing 坐标 `(layer, attn_q|attn_k|attn_v|attn_o_in, index)`，更贴近注意力读写路由。

每个源方法内部仍保持每层/模块最多 top 10%，并且只用 `train` split 发现 A/B/C 共享神经元；TCN 只在这些已经筛好的稀疏神经元空间之间做 circuit-level union。`test` split 只用于后续 probe 报告，不参与神经元发现或训练。

## 输出

```text
../cross_task_tool_neurons_data/probe_prefill/tool_circuit_neurons/probe_features/
../cross_task_tool_neurons_data/probe_prefill/tool_circuit_neurons/probes/
```

核心文件：

```text
train_features.pt
test_features.pt
train_meta.jsonl
test_meta.jsonl
train_summary.json
test_summary.json
manifest.json
```

manifest 参数一致时会提前跳过。若需要清理本方案错误旧产物，在 TCN-1 或 PP-2 原命令末尾追加 `--clean`，只会清理 `tool_circuit_neurons` 对应输出。

## 单卡正式命令

前置要求：已经完成根目录 `README.md` 的阶段 1-3，并且已经完成 `ToolDecisionAnchors`、`ToolKnowledgeNeurons`、`ToolRoutingNeurons` 的 PP-1 特征构建。

TCN-1：融合已完成的源方法 PP-1 特征，不加载生成模型。

```text
python ToolCircuitNeurons/tcn_build_probe_features.py --model-alias qwen3-4b-instruct --source-methods tool_decision_anchors,tool_knowledge_neurons,tool_routing_neurons --subset all
```

TCN-2：沿用 ProbePrefill 原始 logistic probe 训练代码。

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_circuit_neurons --subset all --reg 1000 --max-iter 2000 --threshold 0.5
```
