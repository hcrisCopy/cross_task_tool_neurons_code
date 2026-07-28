# PreciseShield_noABC 单卡运行说明

本目录实现 PreciseShield 的 noABC 版本：不再先找 A/B/C 单类型神经元再取交集，而是直接在 `train` split 上按模型专属 `tool_necessary` 标签做二分类对比，发现一组全任务工具决策神经元。训练和评测不在本目录做，统一交给 `ProbePrefill`。

## 方法思想

神经元定义保持 PreciseShield：

```text
h = sigma(W_gate x + b_gate) * (W_up x + b_up)
n_i^(l) = h_i^(l)
```

代码读取 PreciseShield PS-4 保存的最后输入 token FFN intermediate `h`，并使用 `down_proj` 列范数计算 saliency：

```text
I_i^(l)(D) = |mean_D h_i^(l)| * ||W_down^(l)[:, i]||_2
S_i^(l)(D) = I_i^(l)(D) / (sum_j I_j^(l)(D) + epsilon)
```

noABC 的集合定义为：

```text
D_call   = train 中 tool_necessary=1 的全部样本
D_direct = train 中 tool_necessary=0 的全部样本
PS_noABC_TDN^(l) = TopK_l(S(D_call)) \ TopK_l(S(D_direct))
```

每层取：

```text
k_l = max(floor(p * d_m), min_neurons_per_layer)
```

正式实验默认 `p=0.01`。这里不按 A/B/C 分组，不做 A/B/C 交集；single-hop 和 multi-hop 仍严格分开。

## 前置产物

先完成总 README 的阶段 1-3，得到模型专属标签和改造后数据集。PSNA-4 复用 PreciseShield 的 FFN intermediate activation 产物；如果已经跑过 `PreciseShield/README.md` 的 PS-4，且参数一致，可以直接从 PSNA-5 开始。

## PSNA-4：抽取或复用 PreciseShield 激活

单卡指令：

```text
python PreciseShield/ps_run_activation_extraction.py --model-alias qwen3-4b-instruct --dataset-dir ../cross_task_tool_neurons_data/datasets/modified_when2tool --activations-dir ../cross_task_tool_neurons_data/precise_shield/activations --when2tool-repo third_party/when2tool --subset all --split all --gpus 0 --parallel-mode auto --batch-size 1 --torch-dtype bfloat16 --save-dtype float32 --max-samples 0 --sample-strategy first --seed 2026
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/summary.json
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/manifest.json
```

## PSNA-5：发现 noABC 工具决策神经元

单卡指令：

```text
python PreciseShield_noABC/psna_discover_tool_decision_neurons.py --model-alias qwen3-4b-instruct --activations-dir ../cross_task_tool_neurons_data/precise_shield/activations --neurons-dir ../cross_task_tool_neurons_data/precise_shield_noabc/neurons --visualizations-dir ../cross_task_tool_neurons_data/precise_shield_noabc/visualizations --subset all --intervention-ratio 0.01 --min-neurons-per-layer 1 --heatmap-top-n 300 --epsilon 1.0e-12 --min-class-count 2 --device cuda:0
```

输出：

```text
../cross_task_tool_neurons_data/precise_shield_noabc/neurons/<model_alias>/shared_by_subset/<subset>/PS_noABC_TDN_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield_noabc/neurons/<model_alias>/shared_by_subset/<subset>/saliency_scores.pt
../cross_task_tool_neurons_data/precise_shield_noabc/neurons/<model_alias>/shared_by_subset/<subset>/layer_counts.csv
../cross_task_tool_neurons_data/precise_shield_noabc/neurons/<model_alias>/shared_by_subset/<subset>/class_balance.csv
../cross_task_tool_neurons_data/precise_shield_noabc/neurons/<model_alias>/shared_by_subset/<subset>/summary.json
../cross_task_tool_neurons_data/precise_shield_noabc/neurons/<model_alias>/shared_by_subset/<subset>/manifest.json
../cross_task_tool_neurons_data/precise_shield_noabc/visualizations/<model_alias>/shared_by_subset/*.png
```

打印信息包括 compute device、subset 顺序、逐层 saliency 进度条、每个 subset 的 `PS_noABC_TDN` 数量以及 0/1 标签数量。已完成且 manifest 参数一致会提前跳过，不读取大 activation；发现旧错误产物时在原命令末尾追加：

```text
--clean
```

清理范围只限于 `precise_shield_noabc` 的当前输出。

## 交给 ProbePrefill 训练

PSNA-5 完成后，用 `ProbePrefill` 的公共流程读取 `PS_noABC_TDN` 并训练 logistic probe：

```text
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --probe-method precise_shield_noabc --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy first --require-per-type-labels --seed 2026
```

```text
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --probe-method precise_shield_noabc --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 换模型

只替换命令中的 `--model-alias qwen3-4b-instruct`：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```
