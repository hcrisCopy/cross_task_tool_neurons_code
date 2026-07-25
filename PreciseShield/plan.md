# PreciseShield 版跨任务工具调用神经元实验方案

本文档只规划 `PreciseShield/` 下的新方案。项目总 README 的阶段 1、阶段 2、阶段 3 保持不变：仍然先用 When2Tool 原始数据生成模型专属 `tool_necessary` 标签，再构造模型专属改造后数据集。新方案从“抽取激活”开始替换旧方法，后续神经元探测、热力图、训练和因果验证全部按 PreciseShield 的 FFN 中间神经元定义重写。

## 0. 实验边界

保留不变：

- 阶段 1：原始 When2Tool 数据检查。
- 阶段 2：对每个模型生成 `tool_necessary in {0,1}` 标签。
- 阶段 3：构造模型专属改造后数据集。
- 评测时继续复用 When2Tool 官方 env、tool schema、prompt state machine、final answer 解析和指标口径。

新写且只放在 `PreciseShield/` 下：

- PreciseShield FFN 中间激活提取。
- A/B/C 单类型工具调用神经元探测。
- A/B/C 共享工具调用神经元发现。
- PreciseShield-Masked-LoRA 训练。
- 训练后评测、Base delta、因果验证。

所有产物默认写到仓库同级数据目录：

```text
../cross_task_tool_neurons_data/precise_shield/
|-- activations/
|-- neurons/
|-- visualizations/
|-- checkpoints/
|-- outputs/
|-- causal_validation/
```

这样不会污染旧方案的 `activations/neurons/checkpoints/outputs/causal_validation`。

## 1. 模型和 When2Tool 对齐

模型标签固定为：

```text
qwen3-1.7b
qwen3-4b-instruct
qwen3-14b
qwen3-32b
llama3.1-8b
llama3.3-70b
```

模型路径仍由 `configs/models.yaml` 解析，不在代码里写平台绝对路径。

When2Tool 对齐要求：

- prompt mode：`current`
- reasoning mode：`no_reasoning`
- `enable_thinking=false`
- `max_new_tokens=2048`
- `max_rounds=10` 用于训练/eval 小样本跑通；阶段 2 标签继续按总 README 使用 `max_rounds=12`
- `max_model_len=32768`
- Qwen 系列使用 When2Tool 的 XML tool-call 格式。
- Llama 系列使用 When2Tool 的 native JSON tool-call 格式。
- tool format 不手写判断规则，统一复用旧公共代码对 When2Tool 的 `detect_tool_format` 逻辑。

标签含义继续沿用阶段 2：

```text
tool_necessary = 1: hard_no_tool 模式下答错，说明该样本需要调用工具
tool_necessary = 0: hard_no_tool 模式下答对，说明该样本可以不调工具
```

训练目标是二分类的“第一步调不调工具”，不是训练完整多轮推理能力。

## 2. PreciseShield 神经元定义

PreciseShield 关注 Transformer FFN 的中间神经元。对第 `l` 层 FFN，输入 hidden state 为 `x^(l)`，SwiGLU/GLU 类 FFN 的中间向量写作：

```text
h^(l) = sigma(W_gate^(l) x^(l) + b_gate^(l)) * (W_up^(l) x^(l) + b_up^(l))
y^(l) = W_down^(l) h^(l) + b_down^(l)
```

其中 `*` 是逐元素乘法。第 `l` 层第 `i` 个 PreciseShield 神经元定义为：

```text
n_i^(l) = h_i^(l)
```

也就是说，新方案不再把 `gate_proj/up_proj/down_proj` 的输出通道分别当作三类神经元，而是只把 `down_proj` 输入处的 FFN intermediate coordinate 作为神经元空间。代码中通过 `down_proj` 的 `forward_pre_hook` 捕获该输入张量。

激活抽取位置：

```text
last input token before generation
```

和 When2Tool 的特征抽取思想一致：在模型开始生成第一步动作前，读取最后一个输入 token 的内部状态。

## 3. PreciseShield 重要性分数

对任意样本集合 `D`，第 `l` 层第 `i` 个中间神经元的平均激活为：

```text
abar_i^(l)(D) = mean_{x in D} h_i^(l)(x)
```

`W_down^(l)[:, i]` 是该神经元经过 down projection 影响 residual hidden 维度的列向量。PreciseShield 的神经元重要性写作：

```text
I_i^(l)(D) = || abar_i^(l)(D) * W_down^(l)[:, i] ||_2
```

由于 `abar_i` 是标量，实际计算等价于：

```text
I_i^(l)(D) = |abar_i^(l)(D)| * ||W_down^(l)[:, i]||_2
```

为了在同一层内比较，按层归一化：

```text
S_i^(l)(D) = I_i^(l)(D) / (sum_j I_j^(l)(D) + epsilon)
```

默认 `epsilon=1e-12`。

每层选取 top-k：

```text
k_l = max(floor(p * d_m), min_neurons_per_layer)
```

其中 `d_m` 是 FFN intermediate 维度，`p` 是 PreciseShield 的 intervention ratio。正式实验默认 `p=0.01`；小样本跑通时可临时用 `p=0.05`，避免 A/B/C 交集过小导致训练无法启动。

## 4. 单类型工具调用神经元

对任务类型 `c in {A, B, C}`，在 train split 内分成两个集合：

```text
D_call,c   = {样本 | task_type=c, tool_necessary=1}
D_direct,c = {样本 | task_type=c, tool_necessary=0}
```

这对应 PreciseShield 的“目标能力/安全分布”和“通用分布”差分思想。在本项目里，目标分布是“需要调用工具”的样本，通用分布是“可以直接回答”的样本。

对每一层：

```text
T_call,c^(l)   = TopK_l(S(D_call,c))
T_direct,c^(l) = TopK_l(S(D_direct,c))
PS_TDN_c^(l)   = T_call,c^(l) \ T_direct,c^(l)
```

`PS_TDN_c` 是 PreciseShield 版单任务类型工具决策神经元。集合差分用于过滤“直接回答也同样显著”的通用 FFN 神经元。

最小类别数要求：

```text
min_class_count = 2
```

若某个类型内 `tool_necessary=0/1` 任一类不足，当前 subset 的探测应直接报错，而不是静默改定义。

## 5. 跨类型共享神经元

对单跳和多跳分别计算：

```text
PS_CTD = PS_TDN_A ∩ PS_TDN_B ∩ PS_TDN_C
```

同时输出：

```text
pairwise_AB = PS_TDN_A ∩ PS_TDN_B
pairwise_AC = PS_TDN_A ∩ PS_TDN_C
pairwise_BC = PS_TDN_B ∩ PS_TDN_C
private_A = PS_TDN_A \ PS_CTD
private_B = PS_TDN_B \ PS_CTD
private_C = PS_TDN_C \ PS_CTD
```

`PS_CTD` 是后续训练的唯一神经元集合。`private_*` 只用于因果对照，不用于训练。

## 6. 热力图和统计

阶段 5 画单类型图：

- `ps_density_heatmap_<subset>.png`：每层被选中 PS-TDN 数量占该层 intermediate 维度的比例。
- `ps_tdn_saliency_heatmap_<subset>_<A/B/C>.png`：每个类型 top-N 神经元的 call saliency 分布。
- `layer_counts.csv`：按层统计 PS-TDN 数量。
- `task_type_summary.json`：记录 call/direct 样本数、选中神经元数、top layers。

阶段 6 画共享图：

- `ps_ctd_density_heatmap_<subset>.png`：PS-CTD 按层密度。
- `ps_ctd_saliency_min_heatmap_<subset>.png`：共享神经元在 A/B/C 三类中的最小 saliency。
- `ps_ctd_saliency_mean_heatmap_<subset>.png`：共享神经元在 A/B/C 三类中的平均 saliency。
- `share_rates.csv`：`|PS_CTD| / |PS_TDN_c|`。

热力图只用于观察分布，不反过来改神经元定义。

## 7. 训练设计

训练方法严格对应 PreciseShield 的 masked LoRA：

```text
W' = W0 + (M * B) A
```

其中：

- `A in R^{r x d_in}`
- `B in R^{d_out x r}`
- `M in {0,1}^{d_out}` 是行 mask
- `*` 表示对 LoRA-B 的输出行做 mask

PreciseShield 论文将 mask 加在 FFN `W_up` 和 `W_gate` 的目标 neuron rows。对应本项目：

```text
target modules = gate_proj rows and up_proj rows
target rows    = PS_CTD 的 intermediate neuron indices
```

不训练 `down_proj`，不训练 attention，不训练 embedding，不训练 base 权重。

训练数据：

- 只用 train split。
- 单跳和多跳分开训练两个 adapter。
- `tool_necessary=0`：assistant target 是直接 boxed answer，来自阶段 2 的 hard_no_tool 正确答案记录。
- `tool_necessary=1`：用当前模型在 When2Tool `current/no_reasoning` 下生成第一步动作；只有第一步就是成功 tool call 的样本进入训练。
- 若 `tool_necessary=1` 样本第一步没有成功调工具，记录到 `skipped_examples.jsonl`，不偷偷改成其他目标。

loss：

```text
L = CE(assistant target tokens)
```

prompt、system、tool schema、user instruction token 全部 label 为 `-100`，不参与 loss。

默认训练参数：

```text
rank = 8
lora_alpha = 16
lora_dropout = 0
learning_rate = 5e-5
warmup_ratio = 0.03
max_seq_length = 4096
```

小样本跑通用 `epochs=1` 即可；正式实验建议回到 `epochs=3`。

## 8. 评测指标

阶段 8 只评测训练后模型：

- 只用 test split。
- 加载对应 subset 的 PreciseShield adapter。
- prompt、tool format、state machine 全部走 When2Tool。

阶段 9 评测 Base 并计算 delta：

- Base 不加载 adapter。
- Base 与训练后模型使用同一批 test rows。
- 输出 `comparison_with_base.csv`。

核心指标沿用旧项目和 When2Tool：

- `final_accuracy`
- `tool_calls`
- `avg_tool_calls`
- `tool_call_rate`
- `decision_accuracy`
- `over_call_rate`
- `under_call_rate`
- 每个 task type 的 per-type 指标

因为训练目标是第一步“调不调工具”，`decision_accuracy/over_call/under_call/tool_call_rate` 比最终答案准确率更能反映目标是否被学到。

## 9. 因果验证

因果验证只用 test split，不训练。

干预位置：

```text
down_proj input h
```

对选中 neuron rows，在 forward 时把 `h[..., index]` 置零，作用于所有 token 位置，而不是只作用最后 token。这样验证的是这些 FFN intermediate coordinates 在完整推理/工具调用过程中的因果作用。

干预组：

```text
Base
Mask-Random
Mask-PS-TDN_c
Mask-PS-CTD
Mask-PS-Private_c
```

说明：

- `Base`：不做 mask。
- `Mask-Random`：按 PS-CTD 的每层数量随机抽同层 intermediate neurons。
- `Mask-PS-TDN_c`：对 task type `c`，mask 对应的单类型神经元。
- `Mask-PS-CTD`：mask A/B/C 共享神经元。
- `Mask-PS-Private_c`：对 task type `c`，mask `PS_TDN_c \ PS_CTD`。

预期观察：

- `Mask-PS-CTD` 应对 A/B/C 都产生相对一致的工具调用决策影响。
- `Mask-PS-TDN_c` 应主要影响对应类型。
- `Mask-Random` 应弱于同规模 PS-CTD。
- `Mask-PS-Private_c` 可帮助区分共享和私有机制。

## 10. 跳过与清理

所有阶段都写 `manifest.json`。若输出文件存在且 manifest 参数一致，脚本会提前跳过。

需要清除旧错误产物时，在对应命令末尾加：

```text
--clean
```

清理范围限制在 `../cross_task_tool_neurons_data/precise_shield/` 对应阶段目录内，不碰旧方案产物。

若只是想强制重算但不先清理目录，可用：

```text
--overwrite
```

## 11. qwen3-4b 小样本跑通设置

为了先打通逻辑：

```text
model_alias = qwen3-4b-instruct
single_hop train/test = 100 / 30
multi_hop train/test  = 40 / 30
sample_strategy = balanced
require_per_type_labels = true
```

这保证 A/B/C 三类尽量都有 `tool_necessary=0/1`。如果阶段 5 报某类标签不足，不应改神经元公式，应回到阶段 2/3 增加候选样本或调整抽样。

## 12. 六模型扩展

小样本流程跑通后，对六个模型逐个替换 `--model-alias` 即可。目录结构已按模型隔离：

```text
../cross_task_tool_neurons_data/precise_shield/<stage>/<model_alias>/...
```

单跳、多跳继续分开：

- 神经元探测：`single_hop`、`multi_hop` 分别一套 PS-TDN/PS-CTD。
- 训练：`single_hop`、`multi_hop` 分别一个 adapter。
- 评测和因果验证：分别读取对应 subset 的产物。

不要把不同模型、不同 subset 的神经元或 adapter 混用。
