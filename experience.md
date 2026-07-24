# Experience Notes

本文记录本次打通 `qwen3-4b-instruct` 流程时踩过的坑。重点不是复述最终方案，而是把容易再次犯错的地方写清楚，方便后续扩展到 6 个模型时直接避开。

## 1. 不要手写 When2Tool prompt

最严重的坑是 Stage 4 激活抽取一开始手工拼了 `system + user` messages。这样表面上看起来和 When2Tool 的 `current/no_reasoning` prompt 很像，但实际上没有完全复用官方状态机。

具体遗漏包括：

- `ListManipulationEnv` 的特殊 system prompt。
- 原始样本里可能存在的 `dialog` 字段。
- 官方 `init_state(...)` 里构造的 `messages/tools/envs` 细节。

这个问题会污染 Stage 4 激活，进而影响 Stage 5/6 神经元发现、Stage 7 CTD mask、Stage 9 因果验证。后来已修正为 Stage 4 直接调用：

```text
w2t_utils.init_state(...)
```

再从 state 中取 `messages/tools` 渲染 chat template。

经验：凡是和 When2Tool prompt、tool schema、state machine 有关的地方，优先调用官方函数，不要凭理解手写“等价版本”。

## 2. 跳过机制必须对参数敏感

早期部分阶段只检查产物文件是否存在，容易出现“参数变了但旧产物被复用”的情况。最典型的是：

- Stage 2 manifest 漏记 `tensor_parallel_size`、`max_model_len`、`vllm_dtype`、`record_mode`。
- Stage 4/5/6 主要按文件存在跳过，没有严格比较 manifest。

后来改成：

- Stage 2 记录完整生成相关参数。
- Stage 4 用 `expected_params -> manifest` 精确比较。
- Stage 5 manifest 纳入 Stage 4 activation manifest 参数。
- Stage 6 manifest 纳入 Stage 5 single-type manifest 参数。

经验：只要某阶段依赖上游产物，就要把上游 manifest 的关键参数纳入自己的 manifest。否则上游逻辑修了，下游可能还在吃旧数据。

## 3. Stage 4 修过后，下游必须重跑

Stage 4 从手写 prompt 改为 `init_state(...)` 后，旧激活已经不再是严格对齐版本。因此旧的 Stage 5-9 结果只能作为 smoke 记录，不能作为正式实验结果。

严格实验重跑顺序应是：

```text
Stage 4 -> Stage 5 -> Stage 6 -> Stage 7 -> Stage 8 -> Stage 9 -> Stage 10
```

Stage 2 标签不用因为 Stage 4 prompt 修复而重跑，除非 Stage 2 自己的标签生成参数变了。

经验：一旦改了 prompt 构造、模型输入、activation dtype 或样本选择逻辑，下游发现、训练、因果验证都要视为过期。

## 4. final_report 必须按模型隔离

最初 Stage 10 默认写到：

```text
../cross_task_tool_neurons_data/outputs/final_report/
```

这会导致 6 个模型的最终报告互相覆盖或混在一起。后来改成：

```text
../cross_task_tool_neurons_data/outputs/final_report/<model_alias>/
```

跨模型汇总单独写到：

```text
../cross_task_tool_neurons_data/outputs/final_report/all_models/
```

经验：所有正式产物路径都必须包含 `<model_alias>`，除非这个目录明确表示跨模型汇总。

## 5. 热力图应该在发现阶段生成

曾经误把新增神经元热力图放进 Stage 10 final report。这个位置不对，因为神经元热力图属于发现阶段的解释性产物。

最终修正为：

- Stage 5 同阶段生成单类型神经元热力图和 `TDN-SCAR` 细粒度图。
- Stage 6 同阶段生成共享神经元热力图和 `CTD-SCAR` 细粒度图。
- Stage 10 只汇总已有表格和最终结果图，不重新生成或复制神经元热力图。

经验：可视化跟着它解释的产物走。神经元发现图属于 Stage 5/6，不属于最终汇总阶段。

## 6. activation 保存 dtype 要显式

早期对 Stage 4 激活保存 `float16` 还是 `float32` 有过摇摆。正式实验最后明确：

- 模型前向 dtype 默认 `bfloat16`。
- FFN last-token activation 正式保存 `float32`。
- smoke/debug 可临时用 `float16`，但不能作为正式神经元发现结果。

原因是后续 SCAR 需要均值、方差、z-score 统计，`float32` 更稳，也更接近 When2Tool hidden states `.float()` 的做法。

经验：模型运行 dtype 和保存统计 dtype 是两个参数，必须分开写清楚。

## 7. When2Tool 的轮数参数分阶段不同

容易混淆的点：

- 标签阶段对齐 `extract_features.py`：`max_rounds=12`。
- 训练后评估对齐 `run_eval.py`：`max_rounds=10`。
- `max_new_tokens=2048`、`max_model_len=32768` 要显式写在命令里。
- `tensor_parallel_size` 是硬件并行参数，不是方法变量。

经验：不要把某一个 When2Tool 脚本的默认值套到所有阶段，要分清 `extract_features.py` 和 `run_eval.py` 的口径。

## 8. n_runs=1 只是 smoke，不是正式对齐

Stage 8 一开始默认 `n_runs=1`，这是为了快速走通流程。When2Tool 官方主表是 `n_runs=3` 的 mean/std。

后来补了：

- README 明确 smoke 用 `--n-runs 1`。
- 正式实验显式传 `--n-runs 3`。
- `n_runs>1` 时 `summary_table.csv` 也写 mean/std 扁平表。

经验：默认值可以服务 smoke，但 README 命令必须把 smoke 和正式实验的区别写清楚。

## 9. HFGenerationAgent 是必要实现差异

Stage 7-9 没有完全使用官方 vLLM `AgentModel`，而是使用 `HFGenerationAgent`。这是必要差异，不是方法逻辑偷改：

- Stage 7 要加载自定义 CTD-Masked LoRA。
- Stage 8 要评估训练后的 adapter。
- Stage 9 要注册 activation hook 做 neuron mask。

但 prompt、tool schema、parser、state transition 仍复用 When2Tool 官方代码。

经验：必要实现差异必须写进 README。否则后续对比官方 runtime 时，别人会误以为我们完全复刻了 vLLM 生成栈。

## 10. Qwen3 assistant token mask warning

Stage 7 训练时出现过：

```text
return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.
```

这不是报错。原因是 Qwen3 chat template 不含 `{% generation %}`，Transformers 无法自动返回 assistant token mask。代码实际会 fallback 到 assistant content 子序列匹配。

后来修正为：

- 先检查 chat template 是否包含 `{% generation`。
- 支持才调用 `return_assistant_tokens_mask=True`。
- 不支持就直接 fallback，避免 warning。

经验：不要把 harmless warning 留给用户反复困惑；能在代码里识别并安静 fallback，就应当处理掉。

## 11. 远程验证要小心命令嵌套引号

Windows PowerShell -> ssh -> bash -> Python here-doc 的嵌套引号很容易被吃掉，出现看起来奇怪的错误，比如 Python 字符串引号消失。

经验：

- 长 Python 片段尽量写成临时脚本或简化为直接命令。
- shell 管道和 `grep` pattern 里有 `|` 时要特别小心。
- 验证的核心结果要和命令打印错误区分开，避免把“打印 manifest 失败”误判成“主流程失败”。

## 12. 不要在本地跑模型和污染环境

本项目明确要求本地只写代码，模型运行和测试在远程 `tool_neurons` 环境。实践中本地 Python/NumPy 环境还出现过 Windows MINGW 相关 warning/crash，因此本地只做：

- `compileall`
- `git diff --check`
- 纯文本或轻量 helper 测试

真正涉及模型、tokenizer、When2Tool 环境的验证放到远程。

经验：环境约束不是形式要求。尤其是大模型项目，本地“顺手跑一下”很容易制造不可解释的环境问题。

## 13. README 要记录能复跑的命令

本次多次出现“代码修了，但 README 命令或产物说明还停在旧逻辑”的风险。最终 `tmpREADME.md` 需要同步写清：

- 每个阶段的输入、输出、命令。
- smoke 参数和正式参数的区别。
- 哪些产物会提前跳过。
- 哪些情况需要 `--clean` 或从某阶段开始重跑。

经验：README 不是最后补门面，而是实验可复现性的控制面板。

## 14. 最后保留的原则

后续继续扩展 6 个模型时，优先检查下面几件事：

1. Prompt 是否来自 When2Tool 官方 `init_state/evaluate_batched` 路线。
2. Qwen/XML 与 Llama/native 是否自动分流。
3. 所有输出路径是否带 `<model_alias>`。
4. 所有跳过机制是否比较 manifest 参数。
5. Stage 4 改动后是否重跑 Stage 5-9。
6. smoke 与正式实验参数是否在命令里显式区分。
7. 必要实现差异是否写进 `tmpREADME.md`。

