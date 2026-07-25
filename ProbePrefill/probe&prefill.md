# CTD-Probe&Prefill 实验方案

## 0. 一句话思想

总 README 的阶段 1-6 不改：仍然先用 When2Tool 生成模型专属 `tool_necessary` 标签，抽取 FFN last-token activation，再按 Safety Kernel 的 SCAR 公式分别发现 A/B/C 单类型工具决策神经元，并取交集得到跨任务共享神经元 `CTD`。

从训练阶段开始改掉旧的 CTD-Masked LoRA：不再训练模型权重，而是把 `CTD` 对应的激活坐标拼成特征向量，训练一个二分类线性探针来预测 `tool_necessary`。推理阶段完全模仿 When2Tool 的 Probe&Prefill：探针输出 `P(tool_necessary)`，按阈值 `tau` 选择 soft/hard prefill，然后让原模型继续按官方工具调用流程生成。

本方案回答的问题是：

> Safety Kernel/SCAR 找到的跨任务共享 FFN 神经元，能不能像 When2Tool 的 hidden states 一样，线性解码并控制“该不该调用工具”？

## 1. 与原项目的边界

这次 `ProbePrefill/` 是独立新路线，只读取旧阶段产物，不覆盖旧路线输出。

不改变也不重跑的上游：

- 阶段 1：原始 When2Tool 数据检查。
- 阶段 2：每个模型自己的 `tool_necessary` 标签。
- 阶段 3：模型专属改造后数据集。
- 阶段 4：Safety Kernel 风格 FFN last-token activation。
- 阶段 5：A/B/C 单类型 SCAR 神经元 `TDN_A/TDN_B/TDN_C`。
- 阶段 6：共享神经元 `CTD = TDN_A intersection TDN_B intersection TDN_C`。

ProbePrefill 只从这些路径读取输入：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/<subset>/<split>.jsonl
../cross_task_tool_neurons_data/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/TDN_neurons.jsonl
../cross_task_tool_neurons_data/neurons/<model_alias>/shared_by_subset/<subset>/CTD_neurons.jsonl
```

新路线所有输出统一写入：

```text
../cross_task_tool_neurons_data/probe_prefill/
```

因此不会污染旧的 `checkpoints/outputs/causal_validation`，也不会碰 `PreciseShield/`。

## 2. 固定模型与子集

模型标签固定为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

单跳和多跳严格分开：

| 子集 | 正式 train/test | 小样本打通 train/test | 说明 |
|---|---:|---:|---|
| `single_hop` | 900 / 2250 | 100 / 30 | 训练一个 single-hop CTD probe |
| `multi_hop` | 180 / 450 | 40 / 30 | 训练一个 multi-hop CTD probe |

训练 probe 只用 train；选择阈值曲线和 Probe&Prefill 评测只用 test。这里不在 train 内部再拆 validation，避免和 When2Tool 的 train/test 设定混在一起。阈值 `tau` 是评测时 sweep 的控制旋钮，不用 test 反过来更新 probe 权重。

## 3. 保持不变的神经元定义

沿用总 README 阶段 4-6，也就是 Safety Kernel 风格的 FFN 输出坐标。

对第 `l` 层 FFN 线性模块：

```text
q in {mlp.gate_proj, mlp.up_proj, mlp.down_proj}
```

模块输出为：

```text
a_{l,q}(x) in R^{T_x x d_{l,q}}
```

一个候选神经元是完整坐标：

```text
n = (l, q, i)
```

主实验只取模型开始生成前最后一个输入 token 的模块输出：

```text
h_n(x) = a_{l,q,t_end,i}(x)
```

阶段 5 的 SCAR 公式不改：

```text
delta_n = mu_n^1 - mu_n^0
rho_n = sqrt(((n1-1)(sigma_n^1)^2 + (n0-1)(sigma_n^0)^2) / (n1+n0-2)) + epsilon
D_n = z_{l,q}(delta_n / (rho_n + epsilon))
R_n = z_{l,q}(delta_n)
SCAR_n = D_n + R_n
```

其中 `tool_necessary=1` 类比 unsafe/should-call-tool，`tool_necessary=0` 类比 safe/no-call-tool。A/B/C 分别探测，每类全模型取 `top_k=5000`，再精确按 `(layer, module, index)` 取交集得到：

```text
CTD_{m,s} = TDN_{m,s,A} intersection TDN_{m,s,B} intersection TDN_{m,s,C}
```

ProbePrefill 不重新定义神经元，也不改变 SCAR、top-k、交集规则。

## 4. CTD probe 特征

对固定模型 `m`、子集 `s` 和样本 `x`，读取阶段 6 的共享神经元列表：

```text
CTD_{m,s} = [n_1, n_2, ..., n_K]
```

然后从阶段 4 的 activation 中抽出这些坐标：

```text
phi_CTD(x) = [h_{n_1}(x), h_{n_2}(x), ..., h_{n_K}(x)] in R^K
```

顺序固定为 `CTD_neurons.jsonl` 的 rank 顺序；如果没有 rank，就按 `(layer, module, index)` 稳定排序。训练和测试都使用同一个 `CTD` 坐标集合。

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/probe_features/<model_alias>/<subset>/
|-- train_features.pt
|-- train_meta.jsonl
|-- train_summary.json
|-- train/manifest.json
|-- test_features.pt
|-- test_meta.jsonl
|-- test_summary.json
|-- test/manifest.json
```

`train/test_meta.jsonl` 保存 id、A/B/C、env、difficulty、`tool_necessary`，后续评测必须按这里的 id 顺序取 test 样本，防止 probe 预测和生成评测错位。

## 5. 训练二分类探针

模仿 When2Tool 的 `src/train_probe.py`：

- 特征先用 `StandardScaler` 标准化。
- 分类器用 L2 Logistic Regression。
- `reg=lambda=10000`，即 sklearn `C=1/reg=0.0001`，和 When2Tool 默认主设置一致。
- `max_iter=2000`，`random_state=42`。
- 标签为阶段 2 得到的模型专属 `tool_necessary`。

训练目标：

```text
y_m(x)=1: 这个模型无工具答错，工具必要
y_m(x)=0: 这个模型无工具答对，工具不必要
```

只用 train 拟合 `scaler` 和 `clf`；test 只做报告。探针输出 logit：

```text
z(x) = w^T scaler(phi_CTD(x)) + b
```

评测时按 When2Tool 做温度缩放：

```text
p(x) = sigmoid(z(x) / T)
```

默认 `T=2.0`，阈值 sweep 默认可以用 `tau in {0.1,0.3,0.5,0.7,0.9}`；小样本打通先用 `tau=0.5`。

probe 自身指标：

- AUROC
- Accuracy
- Precision / Recall / F1
- Confusion matrix
- 按 A/B/C、env、difficulty 分组的 AUROC/Accuracy

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/probes/<model_alias>/<subset>/
|-- probe_no_reasoning.pt
|-- probe_results_no_reasoning.json
|-- train_predictions.jsonl
|-- test_predictions.jsonl
|-- probe_coefficients.csv
|-- manifest.json
```

## 6. Probe&Prefill 推理

这一阶段照搬 When2Tool 的 Probe&Prefill 思路，只把 probe 输入从 hidden states 换成 CTD 激活特征。

对每条 test 样本：

1. 读取 `phi_CTD(x)`。
2. 用训练好的 logistic probe 得到 `p=P(tool_necessary)`。
3. 若 `p < tau`，预测不需要工具，使用 no-tool prefill。
4. 若 `p >= tau`，预测需要工具，使用 use-tool prefill。
5. 交给 When2Tool 官方 `evaluate_batched(...)` 正常多轮生成、解析工具调用、执行工具、判断最终答案。

prefill 模板严格按 When2Tool `src/run_probe_eval.py`：

| mode | tool format | `p < tau` | `p >= tau` |
|---|---|---|---|
| soft | xml/native | `I can solve this directly without using a tool.\n` | `I need to use a tool for this question.\n` |
| hard | xml | `\boxed{` | `<tool_call>\n` |
| hard | native | `\boxed{` | `{"name": "` |

默认模式：

- Qwen：`soft`，因为 Qwen 对自然语言 steering 较好。
- Llama：`hard`，因为 When2Tool 发现 Llama 容易忽略 soft prefill。

生成口径：

| 参数 | 值 |
|---|---:|
| prompt mode | `current` |
| reasoning mode | `no_reasoning` |
| enable_thinking | `false` |
| max rounds | 10 |
| max new tokens | 2048 |
| max model len | 32768 |
| n runs | 正式 3，小样本 1 |

tool format 仍按 When2Tool 自动分流：Qwen 走 XML `<tool_call>...</tool_call>`，Llama 走 native JSON tool call + `role=tool` response。每条样本只暴露自己的 env tool schema。

输出按 threshold 隔离：

```text
../cross_task_tool_neurons_data/probe_prefill/outputs/<model_alias>/probe_prefill/<subset>/t0.5_temp2_soft/
|-- outputs.json
|-- per_task.jsonl
|-- summary.json
|-- summary_table.csv
|-- probe_predictions.jsonl
|-- prefills.json
|-- prefill_stats.json
|-- manifest.json
```

## 7. When2Tool 主指标与本实验诊断指标

主指标对齐 When2Tool：

```text
Final Accuracy
Total Tool Calls
Avg Tool Calls
Tool Call Rate
Total Token Cost
Avg Token Cost
```

定义：

```text
Acc = correct / N
TC = sum(tool_calls_i)
AvgTC = TC / N
TCR = sum(tool_calls_i) / sum(expected_steps_i)
TokenCost = generation_tokens + 0.2 * prefill_tokens
```

其中 single-hop 的 `expected_steps=1`，multi-hop 的 `expected_steps=3`。最终答案正确性沿用 When2Tool 的 `item_final_eval`：抽取最后 assistant response 里的 `\boxed{...}`，清洗后和 `expected.answer` 比较。

额外报告工具决策诊断：

```text
actual_call_i = 1[tool_calls_i > 0]
DecisionAcc = mean(actual_call_i == tool_necessary_i)
OverCall = P(actual_call=1 | tool_necessary=0)
UnderCall = P(actual_call=0 | tool_necessary=1)
ToolPrecision / ToolRecall / ToolF1
ValidToolCallRate
ToolTrajectorySuccessRate
```

分组维度：

- overall
- A/B/C task_type
- env
- difficulty
- `tool_necessary=0/1`

## 8. Base/Default 与 delta

为了让 Probe&Prefill 的收益可解释，必须在完全同一批 test 样本上跑 Base/Default：

- 同一模型。
- 同一 subset。
- 同一 test id 列表。
- 同一 prompt/tool/schema/parser/state machine。
- 同一生成参数。
- 不加载 adapter。
- 不做 activation mask。
- 不做 prefill。

Base 输出：

```text
../cross_task_tool_neurons_data/probe_prefill/outputs/<model_alias>/base_evaluation/<subset>/
|-- outputs.json
|-- per_task.jsonl
|-- summary.json
|-- summary_table.csv
|-- manifest.json
```

每个 Probe&Prefill threshold 目录下生成：

```text
comparison_with_base.csv
comparison_with_base_manifest.json
```

delta 指标：

```text
DeltaAcc(pp) = 100 * (Acc_PP - Acc_Base)
DeltaAvgTC = AvgTC_PP - AvgTC_Base
DeltaTCR = TCR_PP - TCR_Base
DeltaTC% = 100 * (TC_PP - TC_Base) / (TC_Base + eps)
ToolCallReduction% = -DeltaTC%
Cost = DeltaAcc(pp) / (-DeltaAvgTC), only if DeltaAvgTC < 0
```

同时报告 `DecisionAcc/OverCall/UnderCall/ToolPrecision/ToolRecall/ToolF1` 的 delta。`Cost` 只在确实节省平均工具调用时有意义，否则记空。

## 9. 因果验证怎么改

因为新方法不再训练 LoRA，因果验证分成两条线。

### 9.1 Probe 特征控制

用同样的 train/test 样本、同样的 Logistic Regression 设置，比较不同神经元特征集合的 probe 可解码性：

- `CTD`
- `Random-CTD`：与 CTD 同层、同模块、同数量随机坐标
- `TDN_A / TDN_B / TDN_C`
- `Private_A / Private_B / Private_C`，其中 `Private_c = TDN_c \ CTD`

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/causal_validation/<model_alias>/<subset>/
|-- probe_control_summary.csv
|-- probe_control_results.json
|-- probe_controls/manifest.json
```

如果 CTD probe 明显优于同规模 random，并且对 A/B/C 分组都有稳定 AUROC，说明共享神经元携带的不是纯随机可分信号。

### 9.2 Activation mask 生成式验证

仍然保留 Safety Kernel 的 activation suppression 思路，在 base 模型生成时把目标 FFN 输出坐标置零，且作用于所有 token 位置：

```text
masked a[..., i] = 0
```

比较：

1. `Base`：不干预。
2. `Mask-Random`：与 CTD 同数量、同层、同模块随机 mask。
3. `Mask-TDN_c`：mask 当前类型的单类型神经元。
4. `Mask-CTD`：mask 共享神经元。
5. `Mask-Private_c`：mask `TDN_c \ CTD`。

输出：

```text
../cross_task_tool_neurons_data/probe_prefill/causal_validation/<model_alias>/<subset>/
|-- random_mask_neurons.jsonl
|-- summary_table.csv
|-- cross_type_summary.csv
|-- activation_mask/manifest.json
|-- A/<intervention>/outputs.json
|-- A/<intervention>/per_task.jsonl
|-- A/<intervention>/summary.json
|-- B/...
|-- C/...
```

主要看：

- `Mask-CTD` 相对 `Base` 是否更一致地改变 A/B/C 的 Acc、TCR、DecisionAcc。
- `Mask-CTD` 是否比 `Mask-Random` 有更强或更稳定的影响。
- `Private_c` 是否主要影响本类型，而 CTD 是否跨类型更一致。

这一步验证的是 CTD 对模型工具行为的功能作用，不再和 LoRA 训练混在一起。

## 10. 完整流程

前置：先按总 README 跑完阶段 1-6。

ProbePrefill 阶段：

1. `pp_build_probe_features.py`：从旧阶段 activation 和 CTD 取 probe 特征。
2. `pp_train_probe.py`：训练 CTD logistic probe。
3. `pp_eval_probe_prefill.py`：按 When2Tool Probe&Prefill 跑 test。
4. `pp_eval_base_and_delta.py`：跑同口径 Base/Default 并计算 delta。
5. `pp_causal_validation.py`：跑 probe 控制和 activation mask 因果验证。

所有脚本都有参数敏感 manifest。输出存在且 manifest 一致会提前跳过；发现旧错误产物时在原命令末尾加 `--clean`，清理范围限制在 `../cross_task_tool_neurons_data/probe_prefill/` 对应阶段内。

## 11. 小样本打通默认

qwen3-4b-instruct 小样本打通假设上游阶段 1-6 已经用以下规模产出：

```text
single-hop train/test = 100 / 30
multi-hop train/test = 40 / 30
覆盖 A/B/C
覆盖 tool_necessary=0/1
```

ProbePrefill 小样本先跑：

```text
threshold = 0.5
temperature = 2.0
prefill_mode = auto  # qwen -> soft, llama -> hard
n_runs = 1
tensor_parallel_size = 1
```

正式实验再把 threshold 改为 `0.1,0.3,0.5,0.7,0.9`，`n_runs=3`，并按实际 GPU 设置 `tensor_parallel_size`。

## 12. 参考实现依据

When2Tool 开源关键点：

- `src/train_probe.py`：`StandardScaler + LogisticRegression`，默认 `reg=10000`，支持 all-layer hidden state。
- `src/run_probe_eval.py`：`threshold`、`temperature`、soft/hard prefill 模板，默认 `threshold=0.7`，论文主表常看 `tau=0.5`。
- `src/utils.py`：`detect_tool_format` 中 Llama 走 native，其余默认 XML；`evaluate_batched` 只在第一轮应用 prefill；`token_cost = generation_tokens + 0.2 * prefill_tokens`。
- `src/run_eval.py`：默认评测 `current/no_reasoning`，`max_rounds=10`，`max_new_tokens=2048`，`max_model_len=32768`，`n_runs=3`。

When2Tool 论文关键点：

- hidden state 是生成前 last-token 表征。
- tool necessity 可被线性 probe 解码，六模型 AUROC 约 0.89-0.96。
- Probe&Prefill 通过阈值 sweep 提供 accuracy/tool-call tradeoff。
- Qwen 适合 soft prefill，Llama 更需要 hard prefill。
- multi-hop 也单独评测；本项目按单跳、多跳分别训练探针，避免混用。

Safety Kernel 关键点：

- 先按条件对比发现单语言/单类型神经元。
- 再取交集得到共享核心。
- top-k 主设置使用 `k=5000`。
- activation suppression 和同规模 random 对照用于证明共享核心不是统计巧合。
