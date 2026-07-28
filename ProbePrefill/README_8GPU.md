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
../cross_task_tool_neurons_data/probe_prefill/precise_shield/
```

`--probe-method safety_kernel` 读取已有 Safety Kernel/CTD 上游产物；如果旧版 ProbePrefill 产物还在根目录，首次运行会非破坏式复制到 `safety_kernel/` 后继续按 manifest 跳过。`--probe-method safety_kernel_union` 读取 `SafetyKernel_Union` 阶段 6 产生的 `CTD_Union`。`--probe-method precise_shield` 读取 PreciseShield 的 PS-4/5/6 产物。

## 运行顺序

推荐按下面 5 条命令依次执行。每条命令内部都会先跑 `single_hop`，再跑 `multi_hop`。

```text
PP-1 -> PP-2 -> PP-3 -> PP-4 -> PP-5
```

PP-5 是因果验证，若本轮只交 Probe&Prefill 主结果，可先跑到 PP-4。

## 上游 SafetyKernel_Union 阶段 6

先按总 README 跑完阶段 1-5，再运行本并集阶段。它只读取阶段 5 的 A/B/C `TDN_neurons.jsonl`，不重新抽 activation，不重新 split。

SafetyKernel_Union / CTD_Union 单卡指令：

```text
python SafetyKernel_Union/sku_discover_union_neurons.py --model-alias qwen3-4b-instruct --input-neurons-dir ../cross_task_tool_neurons_data/neurons --output-neurons-dir ../cross_task_tool_neurons_data/safety_kernel_union/neurons --visualizations-dir ../cross_task_tool_neurons_data/safety_kernel_union/visualizations --subset all --heatmap-top-n 300
```

输出：

```text
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/CTD_Union_neurons.jsonl
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/safety_kernel_union/neurons/<model_alias>/shared_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/safety_kernel_union/visualizations/<model_alias>/shared_by_subset/*.png
```

终端打印每个 subset 的 `CTD_Union` 数量、三类交集数量、两两重叠数量和 membership 分布。产物存在且 manifest 一致会提前跳过；错误旧产物在原命令末尾加 `--clean`。

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

PreciseShield / PS-CTD 单卡指令：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
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

PreciseShield / PS-CTD 单卡指令：

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

终端打印按论文表格版式：`ours` / `when2tool` 两行对比。single-hop 会额外打印 easy/medium/hard AUROC。

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
