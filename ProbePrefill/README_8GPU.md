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
```

`--probe-method safety_kernel` 读取已有 Safety Kernel/CTD 上游产物；如果旧版 ProbePrefill 产物还在根目录，首次运行会非破坏式复制到 `safety_kernel/` 后继续按 manifest 跳过。`--probe-method safety_kernel_union` 读取 `SafetyKernel_Union` 阶段 6 产生的 `CTD_Union`。`--probe-method safety_kernel_noabc` 读取 `SafetyKernel_noABC` 阶段 SKNA-4 激活和 SKNA-5 产生的 `SK_noABC_TDN`。`--probe-method safety_kernel_deepfake` 读取 `SafetyKernel_Deepfake` 阶段 SKD-4 激活和 SKD-6 产生的 `SKD_CTD`。`--probe-method precise_shield` 读取 PreciseShield 的 PS-4/5/6 产物。`--probe-method precise_shield_union` 读取 PreciseShield 阶段 4 激活和 `PreciseShield_Union` 阶段 6 产生的 `PS_CTD_Union`。`--probe-method precise_shield_noabc` 读取 PreciseShield 阶段 4 激活和 `PreciseShield_noABC` 阶段 PSNA-5 产生的 `PS_noABC_TDN`。`--probe-method precise_shield_deepfake` 读取 `PreciseShield_Deepfake` 阶段 PSDF-4 激活和 PSDF-6 产生的 `PSDF_CTD`。
`--probe-method tool_decision_anchors` 读取 `ToolDecisionAnchors` 阶段 TDA-4 激活和 TDA-5 产生的 `TDA_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method residual_decision_anchors` 读取 `ResidualDecisionAnchors` 阶段 RDA-4 residual hidden activation 和 RDA-5 产生的 `RDA_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。
`--probe-method tool_knowledge_neurons` 读取 `ToolKnowledgeNeurons` 阶段 TKN-4 FFN intermediate activation 和 TKN-5 产生的 `TKN_CTD`。后续 PP-1/PP-2 的二分类 probe 训练路径保持不变。

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
