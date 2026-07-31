# ProbePrefill 单机八卡正式运行指令

本文件只给 `python ...` 运行指令。所有路径由项目配置和脚本默认相对路径解析，不写平台路径，不使用额外脚本文件。

核心规则：
- 每个阶段只需要复制一条命令。
- `--subset all` 会在同一条命令里按顺序执行：先 `single_hop`，完成后再 `multi_hop`。
- single-hop 和 multi-hop 的输出目录、summary、delta 仍然分开，不混合统计。
- PP-3、PP-4、PP-5 使用数据并行：`--gpus 0,1,2,3,4,5,6,7` 会启动 8 个 worker，每张卡一个进程、加载一份完整模型、评测约 1/8 的 test 题目。
- `--tensor-parallel-size 1` 保持不变；这里不是 tensor parallel，而是题目维度的数据并行。
- 父进程会打印 GPU 列表、每卡 shard 大小、总题目进度，并汇总 worker 进度条；不使用 wandb。
- 已完成且 manifest 参数一致的产物会提前跳过；错误旧产物可在原命令末尾加 `--clean` 清理本方案输出目录后重跑。

输出根目录：

```text
../cross_task_tool_neurons_data/probe_prefill/
```

不同神经元探测方法的 ProbePrefill 输出统一隔离到子文件夹，方便横向查看和避免互相覆盖：

```text
../cross_task_tool_neurons_data/probe_prefill/safety_kernel/
../cross_task_tool_neurons_data/probe_prefill/safety_kernel_union/
../cross_task_tool_neurons_data/probe_prefill/safety_kernel_noabc/
../cross_task_tool_neurons_data/probe_prefill/safety_kernel_deepfake/
../cross_task_tool_neurons_data/probe_prefill/precise_shield/
../cross_task_tool_neurons_data/probe_prefill/precise_shield_union/
../cross_task_tool_neurons_data/probe_prefill/precise_shield_noabc/
../cross_task_tool_neurons_data/probe_prefill/precise_shield_deepfake/
../cross_task_tool_neurons_data/probe_prefill/tool_decision_anchors/
../cross_task_tool_neurons_data/probe_prefill/residual_decision_anchors/
../cross_task_tool_neurons_data/probe_prefill/tool_knowledge_neurons/
../cross_task_tool_neurons_data/probe_prefill/tkn_activation_gain/
../cross_task_tool_neurons_data/probe_prefill/tool_knowledge_pathways/
../cross_task_tool_neurons_data/probe_prefill/tool_routing_neurons/
../cross_task_tool_neurons_data/probe_prefill/tool_circuit_neurons/
```

`--probe-method safety_kernel` 读取已有 Safety Kernel/CTD 上游产物；如果旧版 ProbePrefill 产物还在根目录，首次运行会非破坏式复制到 `safety_kernel/` 后继续按 manifest 跳过。`--probe-method safety_kernel_union` 读取 `SafetyKernel_Union` 阶段 6 产生的 `CTD_Union`。`--probe-method safety_kernel_noabc` 读取 `SafetyKernel_noABC` 阶段 SKNA-4 激活和 SKNA-5 产生的 `SK_noABC_TDN`。`--probe-method safety_kernel_deepfake` 读取 `SafetyKernel_Deepfake` 阶段 SKD-4 激活和 SKD-6 产生的 `SKD_CTD`。`--probe-method precise_shield` 读取 PreciseShield 的 PS-4/5/6 产物。`--probe-method precise_shield_union` 读取 PreciseShield 阶段 4 激活和 `PreciseShield_Union` 阶段 6 产生的 `PS_CTD_Union`。`--probe-method precise_shield_noabc` 读取 PreciseShield 阶段 4 激活和 `PreciseShield_noABC` 阶段 PSNA-5 产生的 `PS_noABC_TDN`。`--probe-method precise_shield_deepfake` 读取 `PreciseShield_Deepfake` 阶段 PSDF-4 激活和 PSDF-6 产生的 `PSDF_CTD`。
`--probe-method tool_decision_anchors` 读取 `ToolDecisionAnchors` 阶段 TDA-4 激活和 TDA-5 产生的 `TDA_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method residual_decision_anchors` 读取 `ResidualDecisionAnchors` 阶段 RDA-4 residual hidden activation 和 RDA-5 产生的 `RDA_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method tool_knowledge_neurons` 读取 `ToolKnowledgeNeurons` 阶段 TKN-4 FFN intermediate activation 和 TKN-5 产生的 `TKN_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method tkn_activation_gain` 读取 `ToolKnowledgeNeurons/tkn_activation_gain` 直接生成的 `TKN_AG` ProbePrefill 特征；它复用 TKN-4/TKN-5，不重新跑模型，并把 raw activation 变成方向对齐的神经元证据特征，后续 PP-2 的二分类 probe 训练路径保持不变。
`--probe-method tool_knowledge_pathways` 读取 `ToolKnowledgePathways/tkn_pathways` 阶段 TKP-1 产生的 `TKP_TKN_CTD`；它复用 TKN-4/TKN-5 产物，只在 TKN 神经元空间上做方向路径增强，后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method tool_routing_neurons` 读取 `ToolRoutingNeurons` 阶段 TRN-4 attention Q/K/V output、O input activation 和 TRN-5 产生的 `TRN_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method tool_circuit_neurons` 读取 `ToolCircuitNeurons` 阶段 TCN-1 融合后的 `TCN_CTD` ProbePrefill 特征；它不重新抽模型激活，不重跑标签数据，后续 PP-2 的二分类 probe 训练路径保持不变。

## 运行顺序

推荐按下面 5 条命令依次执行。每条命令内部都会先跑 `single_hop`，再跑 `multi_hop`。

```text
PP-1 -> PP-2 -> PP-3 -> PP-4 -> PP-5
```

PP-5 是因果验证，若本轮只交 Probe&Prefill 主结果，可先跑到 PP-4。

## ToolDecisionAnchors 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3。TDA-5 只使用 `train` split 发现神经元，`test` activation 只供 PP-1 构建后续 probe/test 特征。TDA 的神经元定义是 Safety Kernel 风格的 FFN output `(layer, module, index)`，按 A/B/C 三类任务方向一致性计算 shared score，正式命令每个 layer/module 取 top 10%。

TDA-4 单卡指令：
```text
python ToolDecisionAnchors/tda_extract_ffn_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

TDA-5 单卡指令：
```text
python ToolDecisionAnchors/tda_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/activations --neurons-dir ../cross_task_tool_neurons_data/tool_decision_anchors/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_decision_anchors/visualizations --subset all --modules gate_proj,up_proj --top-ratio 0.10 --min-neurons-per-module 1 --min-class-count 2 --heatmap-top-n 300 --epsilon 1.0e-6 --device cuda:0
```

TDA-5 打印每个 subset 的 `TDA_CTD` 数量、score mean/max、modules，并写出 density、全局 top score、逐层 mean score、逐层 top 1% score 热力图。manifest 一致时会提前跳过；需要清理错误旧产物时，在原 TDA-5 命令末尾追加 `--clean`。

## ResidualDecisionAnchors 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3。RDA-4 对齐 When2Tool 的 hidden-state 抽取设置：current prompt、no reasoning、最后输入 token、全层 `outputs.hidden_states`。RDA-5 只使用 `train` split 发现 residual-state 维度神经元，正式命令每层取 top 10%；`test` activation 只供 PP-1 构建后续 probe/test 特征。

RDA-4 单卡指令：
```text
python ResidualDecisionAnchors/rda_extract_hidden_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/activations --when2tool-repo third_party/when2tool --subset all --split all --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --device-map auto --max-samples 0
```

RDA-5 单卡指令：
```text
python ResidualDecisionAnchors/rda_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/activations --neurons-dir ../cross_task_tool_neurons_data/residual_decision_anchors/neurons --visualizations-dir ../cross_task_tool_neurons_data/residual_decision_anchors/visualizations --subset all --top-ratio 0.10 --min-neurons-per-layer 1 --min-class-count 2 --heatmap-top-n 300 --epsilon 1.0e-6 --device cuda:0
```

RDA-5 打印每个 subset 的 `RDA_CTD` 数量、score mean/max，并写出逐层数量图、全局 top score 热力图和逐层 top 1% score 热力图。manifest 一致时会提前跳过；需要清理错误旧产物时，在原 RDA-5 命令末尾追加 `--clean`。

## ToolKnowledgeNeurons 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3。TKN 的神经元定义是 PreciseShield 使用的大模型 FFN intermediate `h`，即每层 MLP `down_proj` 输入处的 `(layer, ffn_intermediate, index)`。TKN-5 只使用 `train` split，按 A/B/C 分别构造 `tool_necessary=1` 与 `tool_necessary=0` 的确定性配对，计算 paired shift、down_proj 列范数加权、方向一致 signed consensus 和最终 `TKN_score`；`test` activation 只供 PP-1 构建后续 probe/test 特征。

TKN-4 单卡指令：
```text
python ToolKnowledgeNeurons/tkn_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 2026
```

TKN-5 单卡指令：
```text
python ToolKnowledgeNeurons/tkn_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/visualizations --subset all --selection top_ratio --top-ratio 0.10 --min-neurons-per-layer 64 --min-shared-score 0.0 --min-pairs 2 --max-pairs 0 --epsilon 1.0e-4 --floor-ratio 0.05 --heatmap-top-n 300 --device cuda:0
```

TKN-5 打印每个 subset 的 `TKN_CTD` 数量、score mean/max、是否使用 down_proj norm，并写出 density、全局 top score、逐层 mean score、逐层 top 1% score 四类热力图。manifest 一致时会提前跳过；需要清理错误旧产物时，在原 TKN-5 命令末尾追加 `--clean`。

## TKNActivationGain / TKN 激活增益前置阶段

该方法复用 TKN-4/TKN-5 已有产物，只使用 `train` split 统计每个 TKN 神经元的方向、阈值和尺度，`test` split 只用冻结好的统计量构建测试特征。它不是把全部激活统一乘常数；统一缩放会被 PP-2 的 `StandardScaler` 抵消。默认 `dual_evidence` 会为每个 TKN 神经元写出 `signed_z`、`positive_gain`、`negative_gain` 三类证据列，让线性 probe 自己组合强工具证据和强直接回答证据。TKAG-1 会直接写出 PP-1 等价的 `train_features.pt/test_features.pt`，所以本方法不再单独运行 `pp_build_probe_features.py`。

TKAG-1 单卡指令：
```text
python ToolKnowledgeNeurons/tkn_activation_gain/tkag_build_probe_features.py --model-alias qwen3-4b-instruct --subset all --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --tkn-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --output-neurons-dir ../cross_task_tool_neurons_data/tkn_activation_gain/neurons --output-probe-root ../cross_task_tool_neurons_data/probe_prefill/tkn_activation_gain --visualizations-dir ../cross_task_tool_neurons_data/tkn_activation_gain/visualizations --keep-ratio 1.0 --gain-lambda 1.0 --evidence-power 2.0 --feature-mode dual_evidence --threshold-mode midpoint --min-neurons-per-layer 1 --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026 --device cuda:0
```

TKAG-1 打印每个 subset 的源 TKN 数量、最终 feature dim、`lambda`、feature mode 和输出路径；写出 `TKAG_CTD_neurons.jsonl`、逐层统计 CSV、三张可视化图，以及 `probe_prefill/tkn_activation_gain/probe_features/` 下的训练/测试特征。manifest 一致时会提前跳过；需要清理错误旧产物时，在原 TKAG-1 命令末尾追加 `--clean`，清理范围只限 `tkn_activation_gain` 本方案输出目录。

## ToolKnowledgePathways / TKN 路径增强前置阶段

该方法复用 TKN-4/TKN-5 已有产物，只使用 `train` split 做路径发现，`test` activation 只供 PP-1 构建后续 probe/test 特征。TKP-1 同时构建 `tool_high` 和 `direct_high` 两条方向路径：先按每层每方向高分 TKN 候选控制规模，再用方向一致 co-activation 建跨 1-4 层跳连，并只对分层短名单边做 sampled masking 因果验证。最终输出保留每层高分 TKN 锚点和路径节点，避免把 TKN 粗暴过滤成很小的断裂图。

TKP-1 单卡指令：
```text
python ToolKnowledgePathways/tkn_pathways/tkp_tkn_discover_pathways.py --model-alias qwen3-4b-instruct --subset all --activations-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/activations --tkn-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --output-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_pathways/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_pathways/visualizations --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --when2tool-repo third_party/when2tool --gpus 0 --device cuda:0 --candidate-per-direction-per-layer 256 --anchor-per-direction-per-layer 96 --final-per-direction-per-layer 192 --max-layer-gap 4 --edge-top-k 4 --activation-quantile 0.70 --min-target-phi 0.02 --generic-penalty 0.5 --min-edge-score -0.01 --causal-mode sampled_mask --causal-sources-per-layer-direction 3 --causal-targets-per-source 4 --causal-samples-per-task 12 --causal-batch-size 2 --min-causal-effect 0.0 --torch-dtype bfloat16 --device-map auto
```

TKP-2 可视化单卡指令：
```text
python ToolKnowledgePathways/tkn_pathways/tkp_tkn_visualize_pathways.py --model-alias qwen3-4b-instruct --subset all --output-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_pathways/neurons --tkn-neurons-dir ../cross_task_tool_neurons_data/tool_knowledge_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_knowledge_pathways/visualizations --top-score-ratio 0.01 --edge-plot-limit 30000 --clean
```

TKP-1 打印每个 subset 的候选规模、coactivation 边数、causal shortlist 数量、最终 `TKP_TKN_CTD` 数量、层覆盖和输出路径；TKP-2 按旧 `ToolKnowledgePathways/tkp_visualize_pathways.py` 的查看习惯写出逐层节点数、逐层 pathway score、TKN top 1% 前后对比热力图和因果边图。manifest 一致时会提前跳过；需要清理错误旧产物时，在原 TKP-1 命令末尾追加 `--clean`，TKP-2 用上面命令里的 `--clean` 只清理本方案可视化目录。

## ToolRoutingNeurons 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3。TRN 的神经元定义采用 Fei Shen 大模型工作中使用过的 attention projection row/column routing coordinate：`q_proj/k_proj/v_proj` 输出坐标和 `o_proj` 输入坐标，即 `(layer, attn_q|attn_k|attn_v|attn_o_in, index)`。TRN-5 只使用 `train` split，按 A/B/C 分别构造 `tool_necessary=1` 与 `tool_necessary=0` 的确定性配对；每个 task type 在每层每个 attention module 内先取 top 10%，再取严格 A/B/C 交集，并用方向一致的 paired-shift shared score 排序。`test` activation 只供 PP-1 构建后续 probe/test 特征。

TRN-4 单卡指令：
```text
python ToolRoutingNeurons/trn_extract_attention_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/tool_routing_neurons/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --device-map auto --max-samples 0
```

TRN-5 单卡指令：
```text
python ToolRoutingNeurons/trn_discover_shared_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/tool_routing_neurons/activations --neurons-dir ../cross_task_tool_neurons_data/tool_routing_neurons/neurons --visualizations-dir ../cross_task_tool_neurons_data/tool_routing_neurons/visualizations --subset all --modules attn_q,attn_k,attn_v,attn_o_in --type-top-ratio 0.10 --min-neurons-per-module 1 --min-shared-score 0.0 --min-consensus-z 0.0 --min-pairs 2 --max-pairs 0 --epsilon 1.0e-4 --floor-ratio 0.05 --support-power 0.5 --heatmap-top-n 300 --device cuda:0
```

TRN-5 打印每个 subset 的 `TRN_CTD` 数量、score mean/max、是否使用 projection norm，并写出 density、全局 top score、逐层 mean score、逐层 top 1% score 四类热力图。manifest 一致时会提前跳过；需要清理错误旧产物时，在原 TRN-5 命令末尾追加 `--clean`。

## ToolCircuitNeurons 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3，并复用 `ToolDecisionAnchors`、`ToolKnowledgeNeurons`、`ToolRoutingNeurons` 已经完成的 PP-1 特征。TCN-1 不加载生成模型，不重新 split，不重跑标签数据；它严格校验 `train/test` 的题目 id 和标签顺序后，将 `TDA_CTD`、`TKN_CTD`、`TRN_CTD` 拼接成 `TCN_CTD`。三个源方法内部仍保持每层/模块最多 top 10%，TCN 只在这些已经筛好的稀疏神经元空间之间做 circuit-level union。

TCN-1 单卡指令：
```text
python ToolCircuitNeurons/tcn_build_probe_features.py --model-alias qwen3-4b-instruct --source-methods tool_decision_anchors,tool_knowledge_neurons,tool_routing_neurons --subset all
```

TCN-1 打印每个 subset/split 的融合 feature dim、样本数和 manifest 跳过信息；需要清理错误旧产物时，在原 TCN-1 命令末尾追加 `--clean`。

## SafetyKernel_Deepfake 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3。完成改造后数据集后，按顺序运行 SKD-4/5/6；只使用 `train` split 发现神经元，`test` activation 只供 PP-1 构建后续评测特征。

SKD-4 单卡指令：
```text
python SafetyKernel_Deepfake/skdf_extract_ffn_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0
```

SKD-5 单卡指令：
```text
python SafetyKernel_Deepfake/skdf_discover_single_type_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/activations --neurons-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations --subset all --top-ratio 0.10 --min-neurons-per-module 1 --heatmap-top-n 300 --epsilon 1.0e-4 --floor-ratio 0.05 --min-pairs 2 --max-pairs 0 --device cuda:0
```

SKD-6 单卡指令：
```text
python SafetyKernel_Deepfake/skdf_discover_shared_neurons.py --model-alias qwen3-4b-instruct --neurons-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_deepfake/visualizations --subset all --heatmap-top-n 300
```

如果 SKD-6 打印某个 subset 的 `SKD_CTD=0`，这是严格 `A/B/C` 交集在当前选择比例下为空；保持方法定义不变，调大 SKD-5 的 `--top-ratio` 后重跑 SKD-5/SKD-6。deepfake-code 默认比例是 `0.10`。

## PreciseShield_Deepfake 前置阶段

该方法复用根目录 `README.md` 的阶段 1-3。完成改造后数据集后，按顺序运行 PSDF-4/5/6；只使用 `train` split 发现神经元，`test` activation 只供 PP-1 构建后续评测特征。神经元空间是 PreciseShield 的 `ffn_intermediate`，分数是 deepfake-code paired shift。

PSDF-4 单卡指令：
```text
python PreciseShield_Deepfake/psdf_extract_intermediate_activations.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 2026
```

PSDF-5 单卡指令：
```text
python PreciseShield_Deepfake/psdf_discover_single_type_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/activations --neurons-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations --subset all --top-ratio 0.10 --min-neurons-per-layer 1 --heatmap-top-n 300 --epsilon 1.0e-4 --floor-ratio 0.05 --min-pairs 2 --max-pairs 0 --device cuda:0
```

PSDF-6 单卡指令：
```text
python PreciseShield_Deepfake/psdf_discover_shared_neurons.py --model-alias qwen3-4b-instruct --neurons-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield_deepfake/visualizations --subset all --heatmap-top-n 300
```

如果 PSDF-6 打印某个 subset 的 `PSDF_CTD=0`，这是严格 `A/B/C` 交集在当前选择比例下为空；保持方法定义不变，调大 PSDF-5 的 `--top-ratio` 后重跑 PSDF-5/PSDF-6。deepfake-code 默认比例是 `0.10`。

## PP-1 构建全量共享神经元 Probe 特征

PP-1 只读取已经完成的 activation、共享神经元和改造后的 train/test 数据，不加载生成模型，不重新 split。`train` 用于训练 probe，`test` 用于后续评测。PP-1 本身没有多卡/单卡差异；下面命令都是单卡正式实验可直接运行的命令。

Safety Kernel / CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method safety_kernel --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

SafetyKernel_Union / CTD_Union 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_union --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

SafetyKernel_noABC / SK_noABC_TDN 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_noabc --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

SafetyKernel_Deepfake / SKD_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_deepfake --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

PreciseShield / PS-CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

PreciseShield_Union / PS-CTD_Union 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield_union --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

PreciseShield_noABC / PS_noABC_TDN 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield_noabc --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

PreciseShield_Deepfake / PSDF_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield_deepfake --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

ToolDecisionAnchors / TDA_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_decision_anchors --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

ResidualDecisionAnchors / RDA_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method residual_decision_anchors --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
```

ToolKnowledgeNeurons / TKN_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_neurons --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

TKNActivationGain / TKN_AG：本方法的 PP-1 等价特征已由 TKAG-1 直接生成，跳过本阶段，不运行 `pp_build_probe_features.py`。

ToolKnowledgePathways / TKP_TKN_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_pathways --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

ToolRoutingNeurons / TRN_CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method tool_routing_neurons --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

ToolCircuitNeurons / TCN_CTD 单卡指令：

```text
python ToolCircuitNeurons/tcn_build_probe_features.py --model-alias qwen3-4b-instruct --source-methods tool_decision_anchors,tool_knowledge_neurons,tool_routing_neurons --subset all
```

## PP-2 训练共享神经元 Logistic Probe

PP-2 只用 train 特征训练 probe；test 只用于报告 AUROC/Accuracy，不参与训练。单跳、多跳会分别训练各自的 probe。

Safety Kernel / CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method safety_kernel --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

SafetyKernel_Union / CTD_Union 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_union --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

SafetyKernel_noABC / SK_noABC_TDN 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_noabc --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

SafetyKernel_Deepfake / SKD_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method safety_kernel_deepfake --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

PreciseShield / PS-CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

PreciseShield_Union / PS-CTD_Union 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield_union --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

PreciseShield_noABC / PS_noABC_TDN 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield_noabc --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

PreciseShield_Deepfake / PSDF_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield_deepfake --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

ToolDecisionAnchors / TDA_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_decision_anchors --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

ResidualDecisionAnchors / RDA_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method residual_decision_anchors --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

ToolKnowledgeNeurons / TKN_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_neurons --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

TKNActivationGain / TKN_AG 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tkn_activation_gain --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

ToolKnowledgePathways / TKP_TKN_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_knowledge_pathways --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

ToolRoutingNeurons / TRN_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_routing_neurons --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

ToolCircuitNeurons / TCN_CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method tool_circuit_neurons --subset all --reg 1000 --max-iter 2000 --threshold 0.5
```

终端打印按论文表格版式：`ours` / `when2tool论文` / `when2tool复现` 对比。single-hop 会额外打印 easy/medium/hard AUROC；复现值目前来自 qwen3-4b-instruct 截图结果。

## PP-3 Probe&Prefill 评测（八卡）

PP-3 只评测 test。每个 subset、每个 tau 下，8 张卡按题目切分并行生成；父进程合并后打印 `Acc / TC / AvgTC` 的 `ours` / `when2tool` 两行对比。

```text
python ProbePrefill/pp_eval_probe_prefill.py --model-alias qwen3-4b-instruct --subset all --thresholds 0.1,0.3,0.5,0.7,0.9 --temperature 2.0 --prefill-mode auto --backend vllm --n-runs 3 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --tensor-parallel-size 1 --vllm-dtype bfloat16 --record-mode lite --seed 2026 --gpus 0,1,2,3,4,5,6,7
```

保存但不在终端主表混排：工具决策诊断、token cost、完整分组、`threshold_sweep_summary.csv`、`threshold_tradeoff.png`。

## PP-4 Base/Default 与 Delta（八卡）

PP-4 先确保同口径 Base/Default test 结果存在；Base 已完成且参数一致时会提前跳过，不重新加载 8 个模型。delta 只读取 Base summary 和 PP-3 summary 计算。

```text
python ProbePrefill/pp_eval_base_and_delta.py --model-alias qwen3-4b-instruct --subset all --thresholds 0.1,0.3,0.5,0.7,0.9 --temperature 2.0 --prefill-mode auto --backend vllm --n-runs 3 --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --tensor-parallel-size 1 --vllm-dtype bfloat16 --record-mode lite --seed 2026 --gpus 0,1,2,3,4,5,6,7
```

终端主表只打印论文 delta 指标：

```text
DeltaAcc(pp), DeltaTC(avg), DeltaAcc/-DeltaTC
```

其中 `DeltaAcc=Acc_PP-Acc_Base`，`DeltaTC(avg)=AvgTC_PP-AvgTC_Base`。single-hop 若包含 `tau=0.5`，会额外打印 easy/medium/hard/overall 分难度对比。

## PP-5 因果验证（八卡）

PP-5 的 probe 控制不加载生成模型；activation-mask 部分使用八卡数据并行。若 activation-mask 已完成且参数一致，会提前跳过 worker 启动。

```text
python ProbePrefill/pp_causal_validation.py --model-alias qwen3-4b-instruct --subset all --reg 10000 --max-iter 2000 --threshold 0.5 --interventions Base,Mask-Random,Mask-TDN_c,Mask-CTD,Mask-Private_c --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 2026 --gpus 0,1,2,3,4,5,6,7
```

## 换模型

只改 `--model-alias`。如果显式传 `--model-path`，必须和 `--model-alias` 对应，禁止把 Qwen3 系列指到 Qwen2.5 的模型目录。

可调但不建议随意改的方法参数：

```text
--thresholds 0.1,0.3,0.5,0.7,0.9
--temperature 2.0
--n-runs 3
--batch-size 1
--max-rounds 10
--max-new-tokens 2048
--max-model-len 32768
--record-mode lite
```

## 重跑与清理

正常重复运行会按 manifest 提前跳过。发现旧产物错误时，在对应原命令末尾追加：

```text
--clean
```

清理范围限制在 ProbePrefill 输出目录内，不会删除阶段 1-6 产物，也不会影响其他方案目录。
