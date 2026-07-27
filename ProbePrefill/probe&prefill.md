# Shared-Neuron Probe&Prefill 实验方案

## 0. 一句话思想

总 README 的阶段 1-3 不改：仍然先用 When2Tool 生成每个模型自己的 `tool_necessary` 标签，再构造模型专属改造后数据集。

从“神经元探测”开始不再固定为某一种方法。ProbePrefill 把训练前阶段抽象成一个可插拔的 `probe_method`：每种探测方法负责产出 train/test activation 和共享神经元列表，ProbePrefill 只负责读取这些神经元坐标，按 When2Tool 的训练方法训练二分类 probe，并在后续评测中照搬 When2Tool 的 Probe&Prefill。

当前已适配两种上游神经元来源：

| `probe_method` | 上游方法 | 共享神经元 | activation 定义 | ProbePrefill 输出 |
|---|---|---|---|---|
| `safety_kernel` | Safety Kernel / SCAR | `CTD` | FFN 线性模块输出 last-token activation | `probe_prefill/safety_kernel/` |
| `precise_shield` | PreciseShield | `PS_CTD` | FFN intermediate `h` last-token activation | `probe_prefill/precise_shield/` |

核心问题从固定的“CTD 能不能解码工具决策”升级为：

> 不同神经元探测方法找到的共享工具决策神经元，能不能像 When2Tool 的 hidden states 一样，线性解码并控制“该不该调用工具”？

## 1. 与原项目的边界

`ProbePrefill/` 是公共训练与评测路线，只读取上游产物，不覆盖上游探测、LoRA 或旧评测输出。

固定不改的公共上游：

- 阶段 1：原始 When2Tool 数据检查。
- 阶段 2：每个模型自己的 `tool_necessary` 标签。
- 阶段 3：模型专属改造后数据集。

方法相关上游：

- Safety Kernel：读取旧阶段 4 activation、阶段 5 A/B/C `TDN`、阶段 6 `CTD`。
- PreciseShield：读取 `PreciseShield/` 的 PS-4 activation、PS-5 A/B/C `PS_TDN`、PS-6 `PS_CTD`。

公共数据输入：

```text
../cross_task_tool_neurons_data/datasets/modified_when2tool/<model_alias>/<subset>/<split>.jsonl
```

Safety Kernel 输入：

```text
../cross_task_tool_neurons_data/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/TDN_neurons.jsonl
../cross_task_tool_neurons_data/neurons/<model_alias>/shared_by_subset/<subset>/CTD_neurons.jsonl
```

PreciseShield 输入：

```text
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/activations.pt
../cross_task_tool_neurons_data/precise_shield/activations/<model_alias>/<subset>/<split>/meta.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/single_type_by_subset/<subset>/<A|B|C>/PS_TDN_neurons.jsonl
../cross_task_tool_neurons_data/precise_shield/neurons/<model_alias>/shared_by_subset/<subset>/PS_CTD_neurons.jsonl
```

ProbePrefill 输出：

```text
../cross_task_tool_neurons_data/probe_prefill/safety_kernel/    # safety_kernel 输出
../cross_task_tool_neurons_data/probe_prefill/precise_shield/   # precise_shield 输出
```

因此新增方法时，优先新增一个方法命名空间，避免不同神经元方案的特征、probe、评测结果互相覆盖。

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
| `single_hop` | 900 / 2250 | 100 / 30 | 训练一个 single-hop shared-neuron probe |
| `multi_hop` | 180 / 450 | 40 / 30 | 训练一个 multi-hop shared-neuron probe |

训练 probe 只用 train；阈值 sweep、Probe&Prefill、Base/delta 只用 test。这里不在 train 内部再拆 validation，避免和 When2Tool 的 train/test 设定混在一起。`--subset all` 只是按顺序跑 `single_hop` 再跑 `multi_hop`，两者不会混合训练或混合统计。

## 3. 方法适配层

ProbePrefill 对每种方法只要求两类信息：

1. activation payload 中的 `module_meta` 和 `activations` 可以按神经元坐标取值。
2. 共享神经元文件中的每行可以映射到 activation 坐标。

### 3.1 Safety Kernel / CTD

Safety Kernel 沿用旧阶段 4-6。对第 `l` 层 FFN 线性模块：

```text
q in {mlp.gate_proj, mlp.up_proj, mlp.down_proj}
a_{l,q}(x) in R^{T_x x d_{l,q}}
n = (l, q, i)
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

其中 `tool_necessary=1` 类比 should-call-tool，`tool_necessary=0` 类比 no-call-tool。A/B/C 分别探测，每类全模型取 `top_k=5000`，再精确按 `(layer, module, index)` 取交集：

```text
CTD = TDN_A intersection TDN_B intersection TDN_C
```

### 3.2 PreciseShield / PS-CTD

PreciseShield 使用 FFN intermediate：

```text
h = sigma(W_gate x + b_gate) * (W_up x + b_up)
n_i^(l) = h_i^(l)
```

代码通过 `down_proj` 的 `forward_pre_hook` 抓取 `down_proj` 输入处的 `h`，取模型开始生成前最后一个输入 token。对样本集合 `D`：

```text
I_i^(l)(D) = |mean_D h_i^(l)| * ||W_down^(l)[:, i]||_2
S_i^(l)(D) = I_i^(l)(D) / (sum_j I_j^(l)(D) + epsilon)
```

每层取：

```text
k_l = max(floor(p * d_m), min_neurons_per_layer)
```

A/B/C 单类型：

```text
D_call,c   = task_type=c and tool_necessary=1
D_direct,c = task_type=c and tool_necessary=0
PS_TDN_c^(l) = TopK_l(S(D_call,c)) \ TopK_l(S(D_direct,c))
```

共享神经元：

```text
PS_CTD = PS_TDN_A intersection PS_TDN_B intersection PS_TDN_C
```

ProbePrefill 不改 PreciseShield 的探测公式，只读取 `PS_CTD_neurons.jsonl` 并抽取对应 intermediate activation。

## 4. 共享神经元 probe 特征

对固定模型 `m`、子集 `s`、方法 `r` 和样本 `x`，读取方法 `r` 的共享神经元：

```text
N_{r,m,s} = [n_1, n_2, ..., n_K]
```

然后从对应 activation 中抽坐标：

```text
phi_r(x) = [h_{n_1}(x), h_{n_2}(x), ..., h_{n_K}(x)] in R^K
```

顺序固定为共享神经元文件的 `rank` 顺序；如果没有 rank，就按稳定坐标排序。训练和测试都使用同一个共享神经元坐标集合。

Safety Kernel 输出：

```text
../cross_task_tool_neurons_data/probe_prefill/safety_kernel/probe_features/<model_alias>/<subset>/
```

PreciseShield 输出：

```text
../cross_task_tool_neurons_data/probe_prefill/precise_shield/probe_features/<model_alias>/<subset>/
```

每个目录包含：

```text
train_features.pt
train_meta.jsonl
train_summary.json
train/manifest.json
test_features.pt
test_meta.jsonl
test_summary.json
test/manifest.json
```

`train/test_meta.jsonl` 保存 id、A/B/C、env、difficulty、`tool_necessary`。后续评测必须按这里的 id 顺序取 test 样本，防止 probe 预测和生成评测错位。

## 5. 训练二分类探针

不同方案的核心差异不是分类器，而是送进 probe 的特征向量。统一记为：

```text
u_m,s,r(x) in R^d
```

其中 `m` 是模型，`s` 是 `single_hop` 或 `multi_hop`，`r` 是特征来源。所有方案都用同一个模型专属 `tool_necessary` 作为标签，只用 train 拟合 probe，test 只报告和后续评测。

| 方案 | probe 输入怎么构造 | 训练方式 |
|---|---|---|
| When2Tool 原始 hidden-state probe | 对每条样本做一次模型前向，取开始生成前最后一个输入 token；收集所有 Transformer 层的 hidden state `H_l(x)[t_end]`，按层顺序拼接成 `[H_1; H_2; ...; H_L]`，作为全层 hidden states 特征。 | `StandardScaler + L2 LogisticRegression` 训练二分类 probe，标签是该模型自己的 `tool_necessary`。 |
| Safety Kernel / CTD probe | 先由 Safety Kernel/SCAR 在 A/B/C 中挖掘 `TDN`，再取交集得到 `CTD`。ProbePrefill 读取 `CTD_neurons.jsonl`，从 stage4 保存的 FFN 线性模块输出 activation 中，取最后输入 token 上这些 `(layer, module, index)` 坐标的值，按 CTD 顺序拼成 `phi_safety_kernel(x)`。 | 分类器完全照搬 When2Tool；只是输入从全层 hidden states 换成 CTD 神经元 activation。 |
| PreciseShield / PS-CTD probe | 先由 PreciseShield 根据 FFN intermediate `h` 的重要性得分挖掘 A/B/C 的 `PS_TDN`，再取交集得到 `PS_CTD`。ProbePrefill 读取 `PS_CTD_neurons.jsonl`，从 PS-4 保存的 `down_proj` 输入处 intermediate `h` 中，取最后输入 token 上这些 `(layer, index)` 坐标的值，按 PS-CTD 顺序拼成 `phi_precise_shield(x)`。 | 分类器完全照搬 When2Tool；只是输入换成 PS-CTD intermediate 神经元 activation。 |
| 后续新增探测方法 | 新方法只需要产出 train/test activation 和共享神经元列表，并在 `ProbePrefill` 中定义如何由神经元坐标抽出 `phi_r(x)`。 | 继续复用同一套 `StandardScaler + L2 LogisticRegression`、同一套 test 评测和 Probe&Prefill。 |

因此本项目的比较口径是固定的：When2Tool 原始方法用“last token 拼接全层 hidden states”，本方案系列用“last token 上被不同神经元探测方法选中的神经元 activation”。训练器、标签、train/test 划分和后续阈值评测保持一致。

训练实现照搬 When2Tool 的 `src/train_probe.py`：

- 特征先用 `StandardScaler` 标准化。
- 分类器用 L2 Logistic Regression。
- `reg=lambda=10000`，即 sklearn `C=1/reg=0.0001`。
- `max_iter=2000`，`random_state=42`。
- 标签为阶段 2 得到的模型专属 `tool_necessary`。

训练目标：

```text
y_m(x)=1: 这个模型无工具答错，工具必要
y_m(x)=0: 这个模型无工具答对，工具不必要
```

只用 train 拟合 `scaler` 和 `clf`；test 只做报告。探针输出 logit：

```text
z(x) = w^T scaler(phi_r(x)) + b
```

评测时按 When2Tool 做温度缩放：

```text
p(x) = sigmoid(z(x) / T)
```

默认 `T=2.0`，阈值 sweep 默认使用：

```text
tau in {0.1,0.3,0.5,0.7,0.9}
```

Probe 输出目录与方法命名空间一致：

```text
probes/<model_alias>/<subset>/
|-- probe_no_reasoning.pt
|-- probe_results_no_reasoning.json
|-- train_predictions.jsonl
|-- test_predictions.jsonl
|-- probe_coefficients.csv
|-- manifest.json
```

## 6. Probe&Prefill 推理

这一阶段照搬 When2Tool 的 Probe&Prefill 思路，只把 probe 输入从 hidden states 换成 `phi_r(x)`。

对每条 test 样本：

1. 读取 `phi_r(x)`。
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

## 7. When2Tool 主指标与诊断指标

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

delta 指标：

```text
DeltaAcc(pp) = 100 * (Acc_PP - Acc_Base)
DeltaAvgTC = AvgTC_PP - AvgTC_Base
DeltaTCR = TCR_PP - TCR_Base
DeltaTC% = 100 * (TC_PP - TC_Base) / (TC_Base + eps)
ToolCallReduction% = -DeltaTC%
Cost = DeltaAcc(pp) / (-DeltaAvgTC), only if DeltaAvgTC < 0
```

## 9. 因果验证定位

当前 `pp_causal_validation.py` 仍是 Safety Kernel / CTD 的因果验证入口，包含：

- probe 控制：比较 `CTD`、同规模 `Random-CTD`、`TDN_A/B/C`、`Private_A/B/C`。
- activation mask：在生成时把 FFN 输出坐标置零，比较 `Mask-CTD` 和 random/private/TDN 对 Acc、TCR、DecisionAcc 的影响。

PreciseShield 的 generation-time mask 作用在 `down_proj` 输入 `h`，机制不同，已经由 `PreciseShield/` 自己的 PS-10 实现。ProbePrefill 目前先公共化 PP-1 到 PP-4；后续若要把 PS 因果验证也纳入 PP 公共入口，需要新增 intermediate mask adapter，而不是复用 Safety Kernel 的 FFN 输出 mask。

## 10. 完整流程

前置：

1. 先按总 README 跑完阶段 1-3。
2. 选择神经元探测方法并跑完其 activation 和共享神经元阶段：
   - Safety Kernel：旧阶段 4-6。
   - PreciseShield：PS-4 到 PS-6。

ProbePrefill 阶段：

1. `pp_build_probe_features.py`：按 `--probe-method` 从对应 activation 和共享神经元取 probe 特征。
2. `pp_train_probe.py`：训练 shared-neuron logistic probe。
3. `pp_eval_probe_prefill.py`：按 When2Tool Probe&Prefill 跑 test。
4. `pp_eval_base_and_delta.py`：跑同口径 Base/Default 并计算 delta。
5. `pp_causal_validation.py`：当前仅用于 Safety Kernel / CTD 因果验证。

所有脚本都有参数敏感 manifest。输出存在且 manifest 一致会提前跳过；发现旧错误产物时在原命令末尾加 `--clean`，清理范围限制在对应 ProbePrefill 输出命名空间内。

## 11. 正式命令入口

正式命令维护在：

```text
ProbePrefill/README_8GPU.md
```

当前 PP-1 和 PP-2 已分别给出 Safety Kernel 与 PreciseShield 的单卡正式命令。PP-3/PP-4 的评测入口已经支持 `--probe-method`，但 README 暂时保留八卡主评测命令，后续正式跑 PreciseShield Probe&Prefill 评测时再补对应评测阶段命令。

## 12. 参考实现依据

When2Tool 开源关键点：

- `src/train_probe.py`：`StandardScaler + LogisticRegression`，支持 all-layer hidden state。
- `src/run_probe_eval.py`：`threshold`、`temperature`、soft/hard prefill 模板。
- `src/utils.py`：Qwen 走 XML，Llama 走 native；`evaluate_batched` 只在第一轮应用 prefill；`token_cost = generation_tokens + 0.2 * prefill_tokens`。
- `src/run_eval.py`：默认评测 `current/no_reasoning`，`max_rounds=10`，`max_new_tokens=2048`，`max_model_len=32768`，`n_runs=3`。

When2Tool 论文关键点：

- hidden state 是生成前 last-token 表征。
- tool necessity 可被线性 probe 解码，六模型 AUROC 约 0.89-0.96。
- Probe&Prefill 通过阈值 sweep 提供 accuracy/tool-call tradeoff。
- Qwen 适合 soft prefill，Llama 更需要 hard prefill。
- multi-hop 单独评测，本项目也按单跳、多跳分别训练 probe。

Safety Kernel 关键点：

- 先按条件对比发现单类型神经元。
- 再取交集得到共享核心。
- top-k 主设置使用 `k=5000`。
- activation suppression 和同规模 random 对照用于证明共享核心不是统计巧合。

PreciseShield 关键点：

- 神经元定义为 FFN intermediate `h`，不是 FFN 线性模块输出。
- saliency 同时考虑平均激活强度和 `W_down` 下游影响。
- 通过 `TopK(call) \ TopK(direct)` 去掉通用神经元，再跨 A/B/C 取 `PS_CTD`。
- targeted tuning/mask 作用位置和 Safety Kernel 不同，因此因果验证需要独立 adapter。
