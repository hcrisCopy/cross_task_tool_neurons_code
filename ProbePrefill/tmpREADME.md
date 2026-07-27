# ProbePrefill 小样本打通记录

本目录是新方案入口，只做“CTD 激活特征 -> 二分类 probe -> Probe&Prefill -> Base/delta -> 因果验证”。总 README 的阶段 1-6 保持不变，本目录只读取它们的产物，不重跑、不覆盖。

本次打通模型：

```text
qwen3-4b-instruct
```

小样本规模：

```text
single_hop: train 100, test 30
multi_hop:  train 40,  test 30
要求覆盖 A/B/C，覆盖 tool_necessary=0/1
```

所有新输出：

```text
../cross_task_tool_neurons_data/probe_prefill/
```

## 文件说明

```text
ProbePrefill/probe&prefill.md              # 新实验方案
ProbePrefill/pp_common.py                  # 公共路径、CTD 特征、prefill 模板、指标和 manifest
ProbePrefill/pp_reporting.py               # 指标解释、When2Tool 论文参考值、表格和图片报告
ProbePrefill/pp_build_probe_features.py    # PP-1：从旧阶段 activation + CTD 构建 probe 特征
ProbePrefill/pp_train_probe.py             # PP-2：训练 CTD logistic probe
ProbePrefill/pp_eval_probe_prefill.py      # PP-3：照搬 When2Tool Probe&Prefill 评测
ProbePrefill/pp_eval_base_and_delta.py     # PP-4：同口径 Base/Default 和 delta
ProbePrefill/pp_causal_validation.py       # PP-5：probe 控制 + activation mask 因果验证
ProbePrefill/tmpREADME.md                  # 本文件
```

## 前置条件

先确认总 README 阶段 1-6 已经完成，且这些文件存在：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/single_hop/train.jsonl
../cross_task_tool_neurons_data/datasets/modified_when2tool/qwen3-4b-instruct/multi_hop/train.jsonl
../cross_task_tool_neurons_data/activations/qwen3-4b-instruct/single_hop/train/activations.pt
../cross_task_tool_neurons_data/activations/qwen3-4b-instruct/multi_hop/train/activations.pt
../cross_task_tool_neurons_data/neurons/qwen3-4b-instruct/shared_by_subset/single_hop/CTD_neurons.jsonl
../cross_task_tool_neurons_data/neurons/qwen3-4b-instruct/shared_by_subset/multi_hop/CTD_neurons.jsonl
```

如果上游产物不存在，先回总 README 或根目录 `tmpREADME.md` 跑阶段 1-6。本目录不负责生成标签、改造数据、抽 activation 或发现 CTD。

## PP-1 构建 CTD probe 特征

single-hop：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --subset single_hop --max-train-samples 100 --max-test-samples 30 --sample-strategy balanced --require-per-type-labels --seed 2026
```

multi-hop：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --subset multi_hop --max-train-samples 40 --max-test-samples 30 --sample-strategy balanced --require-per-type-labels --seed 2026
```

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/probe_features/qwen3-4b-instruct/<subset>/
|-- train_features.pt
|-- train_meta.jsonl
|-- train_summary.json
|-- train/manifest.json
|-- test_features.pt
|-- test_meta.jsonl
|-- test_summary.json
|-- test/manifest.json
```

做法：按 Stage 6 的 `CTD_neurons.jsonl` 顺序，从 Stage 4 的 last-token FFN activation 里抽对应坐标，拼成 `phi_CTD(x)`。train/test 分开写；不会重新 split。

## PP-2 训练 CTD 二分类 probe

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/probes/qwen3-4b-instruct/<subset>/
|-- probe_no_reasoning.pt
|-- probe_results_no_reasoning.json
|-- train_predictions.jsonl
|-- test_predictions.jsonl
|-- probe_coefficients.csv
|-- probe_metrics_table.csv
|-- probe_report.md
|-- probe_roc_curve.png
|-- probe_probability_hist.png
|-- manifest.json
```

做法：`StandardScaler + L2 LogisticRegression`，只用 train 拟合；test 只报告 AUROC、Accuracy、Precision/Recall/F1 和按 A/B/C、env、difficulty 的分组结果。

关键点：
- `AUROC/Accuracy` 是当前 CTD 神经元特征 probe 的结果。
- 报告里同时打印 When2Tool 论文 hidden-state probe 参考值，只用于对照，不是本实验结果。
- `threshold` 只影响二分类报告和后续 prefill 决策，不会反过来训练 probe。

## PP-3 Probe&Prefill 评测

```text
python ProbePrefill/pp_eval_probe_prefill.py --model-alias qwen3-4b-instruct --subset all --thresholds 0.5 --temperature 2.0 --prefill-mode auto --backend vllm --tensor-parallel-size 1 --n-runs 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --record-mode lite
```

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/outputs/qwen3-4b-instruct/probe_prefill/<subset>/t0.5_temp2_soft/
|-- outputs.json
|-- per_task.jsonl
|-- summary.json
|-- summary_table.csv
|-- probe_predictions.jsonl
|-- prefills.json
|-- prefill_stats.json
|-- metric_report.md
|-- manifest.json

../cross_task_tool_neurons_data/probe_prefill/outputs/qwen3-4b-instruct/probe_prefill/<subset>/
|-- threshold_sweep_summary.csv
|-- threshold_sweep_report.md
|-- threshold_tradeoff.png
```

做法：读取 test 特征，用 probe 得到 `P(tool_necessary)`；`p < tau` 填 “I can solve this directly without using a tool.”，否则填 “I need to use a tool for this question.”。qwen 自动用 soft prefill；Llama 会自动用 hard prefill。评测走 When2Tool 官方 `evaluate_batched`。

打印指标：
- 主指标：Final Accuracy、Total Tool Calls、Avg Tool Calls、Tool Call Rate、Total/Avg Token Cost。
- 诊断指标：DecisionAcc、OverCall、UnderCall、ToolPrecision/Recall/F1、ValidToolCallRate、ToolTrajectorySuccessRate。
- `Tool Call Rate = sum(tool_calls) / sum(expected_steps)`；single-hop 通常 `expected_steps=1`，multi-hop 通常 `expected_steps=3`。
- `Token Cost = generation_tokens + 0.2 * prefill_tokens`，严格跟 When2Tool 代码一致。

## PP-4 Base/Default 和 delta

```text
python ProbePrefill/pp_eval_base_and_delta.py --model-alias qwen3-4b-instruct --subset all --thresholds 0.5 --temperature 2.0 --prefill-mode auto --backend vllm --tensor-parallel-size 1 --n-runs 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --record-mode lite
```

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/outputs/qwen3-4b-instruct/base_evaluation/<subset>/
|-- outputs.json
|-- per_task.jsonl
|-- summary.json
|-- summary_table.csv
|-- manifest.json
|-- metric_report.md

../cross_task_tool_neurons_data/probe_prefill/outputs/qwen3-4b-instruct/probe_prefill/<subset>/t0.5_temp2_soft/
|-- comparison_with_base.csv
|-- comparison_with_base_manifest.json

../cross_task_tool_neurons_data/probe_prefill/outputs/qwen3-4b-instruct/probe_prefill/<subset>/
|-- delta_sweep_summary.csv
|-- delta_report.md
|-- delta_tradeoff.png
```

做法：Base 不加 prefill、不加 adapter、不做 mask，其他 prompt/tool/schema/parser/生成参数和 PP-3 一致。随后按同分组计算 `DeltaAcc(pp)`、`DeltaAvgTC`、`DeltaTC%`、`ToolCallReduction%`、`Cost` 和工具决策诊断 delta。

delta 口径：
- 全部 delta 都是 `CTD-Probe&Prefill - Base/Default`，同模型、同 subset、同一批 test id、同一套 When2Tool parser/state machine。
- `DeltaAcc(pp)=100*(Acc_PP-Acc_Base)`，单位是百分点。
- `DeltaAvgTC=AvgTC_PP-AvgTC_Base`，负数表示少调用工具。
- `ToolCallReduction%=-100*(TC_PP-TC_Base)/(TC_Base+eps)`，正数表示相对 Base 节省工具调用。
- `Cost=DeltaAcc(pp)/(-DeltaAvgTC)`，只有 `DeltaAvgTC<0` 时才有意义。

## PP-5 因果验证

完整小样本命令：

```text
python ProbePrefill/pp_causal_validation.py --model-alias qwen3-4b-instruct --subset all --reg 10000 --threshold 0.5 --interventions Base,Mask-Random,Mask-TDN_c,Mask-CTD,Mask-Private_c --batch-size 1 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --torch-dtype bfloat16 --device-map auto --record-mode lite --seed 2026
```

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/causal_validation/qwen3-4b-instruct/<subset>/
|-- probe_control_summary.csv
|-- probe_control_results.json
|-- probe_control_report.md
|-- probe_control_bars.png
|-- probe_controls/manifest.json
|-- random_mask_neurons.jsonl
|-- summary_table.csv
|-- cross_type_summary.csv
|-- activation_mask_report.md
|-- activation_mask_delta.png
|-- activation_mask/manifest.json
|-- A/<intervention>/outputs.json
|-- A/<intervention>/per_task.jsonl
|-- A/<intervention>/summary.json
|-- B/...
|-- C/...
```

做法：

- probe 控制：比较 `CTD`、同规模 `Random-CTD`、`TDN_A/B/C`、`Private_A/B/C` 特征训练出来的 probe AUROC/Accuracy。
- activation mask：用未训练 base 模型，在生成时把对应 FFN 输出坐标置零，比较 `Mask-CTD` 和 random/private/TDN 对 Acc、TCR、DecisionAcc 的影响。

如果只想先检查 probe 控制逻辑，不跑生成式 mask，可临时加：

```text
--skip-activation-mask
```

## 清理旧错误产物

每个阶段都支持提前跳过：产物存在且 manifest 参数一致就不重跑。

发现旧产物错了，在原命令末尾加：

```text
--clean
```

清理范围限制在：

```text
../cross_task_tool_neurons_data/probe_prefill/
```

不会删旧方案阶段 1-6，也不会删 PreciseShield。

## 正式实验怎么改

把 PP-1 的小样本参数去掉或设成 0：

```text
--max-train-samples 0 --max-test-samples 0
```

把 PP-3 和 PP-4 改成阈值 sweep 和 3 次评测：

```text
--thresholds 0.1,0.3,0.5,0.7,0.9 --n-runs 3
```

单机八卡评测小/中模型时，PP-3、PP-4、PP-5 命令追加：

```text
--gpus 0,1,2,3,4,5,6,7 --tensor-parallel-size 1
```

这里是按题目切 shard 的数据并行，每张卡独立加载一份模型并评测一部分 test 样本；不是方法变量。现在单卡打通就保持：

```text
--gpus 0 --tensor-parallel-size 1
```

切换模型只改：

```text
--model-alias qwen3-4b-instruct
```

替换为六个模型标签之一。`tensor_parallel_size` 只按硬件调，不作为方法变量。
