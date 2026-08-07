# GitHub Issue 草稿（英文，可直接粘贴）

> 仓库: https://github.com/Trustworthy-ML-Lab/when2tool
> 粘贴前只需补充 3 个占位符：OS 发行版 / GPU 型号 / Python 版本（`__FILL__`）

---

## Title

**[Reproduction] Multi-hop probe metrics mismatch for Qwen3-4B-Instruct (0.9257/0.8489 vs paper 0.966/0.947) — env/data/model all match; could you share the ground-truth labels?**

## Body

Hi, thanks for the great work! I'm trying to reproduce the multi-hop probe numbers in Table 10 for `qwen3-4b-instruct`. I ran the official pipeline with no code changes, and with the same data, model and dependency versions as the repo, but the probe metrics differ notably from the paper:

| Setting (multi-hop, no_reasoning, all-layers, reg=10000) | AUROC | Accuracy |
|---|---|---|
| Paper (Table 10) | 0.966 | 0.947 |
| My reproduction | 0.9257 | 0.8489 |

### Environment

- torch 2.6.0+cu124 / transformers 4.55.2 / vllm 0.8.5 / accelerate 1.13.0 / scikit-learn 1.7.2 — exactly matches the pinned `requirements.txt`
- OS: `__FILL__` | GPU: `__FILL__` | Python: `__FILL__`

### Data & model (same as the repo)

- **Data**: generated from `cesun/When2Tool` via `datasets.load_dataset`, using the same field-reconstruction logic as `src/utils.py::_load_from_hf` (single_hop train/test 900/2250, multi_hop train/test 180/450). No modification.
- **Model**: `Qwen/Qwen3-4B-Instruct-2507` (downloaded via ModelScope, matching the alias `qwen3-4b-instruct` in `MODEL_NAME_MAP`).

### Steps to reproduce

Step 1 — generate data (same logic as `utils._load_from_hf`):

```python
python - <<'PY'
import json
from datasets import load_dataset

def dump(config, split, path):
    ds = load_dataset("cesun/When2Tool", config, split=split)
    tasks = []
    for row in ds:
        task = {
            "id": row["id"],
            "difficulty": row["difficulty"],
            "multi_step": row["multi_step"],
            "instruction": row["instruction"],
            "environments": [{
                "name": row["env_name"],
                "tools": json.loads(row["tools"]),
                "parameters": json.loads(row["parameters"]),
            }],
            "expected": {"answer": row["answer"]},
            "tags": json.loads(row["tags"]),
        }
        steps = json.loads(row["steps"])
        if steps:
            task["expected"]["steps"] = steps
        tasks.append(task)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(path, len(tasks))

dump("single_hop", "train", "data/tasks_v1_train.json")
dump("single_hop", "test", "data/tasks_v1_test.json")
dump("multi_hop", "train", "data/tasks_v1_multihop_train.json")
dump("multi_hop", "test", "data/tasks_v1_multihop_test.json")
PY
```

Step 2 — extract features (equivalent to step 2 of `run_multihop_pipeline.sh`; other args use script defaults `phase=all`, `max_new_tokens=2048`, `max_rounds=12`):

```bash
CUDA_VISIBLE_DEVICES=0 python src/extract_features.py \
  --model_path qwen3-4b-instruct \
  --output_dir ./probe_data/qwen3-4b-instruct_multihop \
  --data_path ./data/tasks_v1_multihop_train.json \
  --data_path_test ./data/tasks_v1_multihop_test.json \
  --tensor_parallel_size 1 \
  --max_model_len 32768
```

Step 3 — train probe (identical to step 3 of the pipeline):

```bash
python src/train_probe.py \
  --data_dir ./probe_data/qwen3-4b-instruct_multihop \
  --mode no_reasoning \
  --reg 10000 \
  --all_layers
```

### What I checked / ruled out

1. ~~Command/settings~~: identical to the official pipeline (only `tensor_parallel_size=1` for speed).
2. ~~Dependencies~~: exact match with `requirements.txt`.
3. ~~Data~~: same HF dataset, same reconstruction logic.
4. ~~Model revision~~: same `Qwen/Qwen3-4B-Instruct-2507` snapshot.

### Remaining suspect: label randomness

The `tool_necessary` labels come from the vLLM `hard_no_tool` evaluation (`evaluate_batched` in `src/utils.py`). Sampling uses the model's own `generation_config` (`VLLMAgentBackend._sampling_params` in `src/model.py`); Qwen3-4B-Instruct's defaults are temperature=0.7, top_p=0.8, top_k=20, and I could not find any fixed seed — neither in the `vLLM(...)` init nor in `SamplingParams`. So the label set itself changes between runs, which would shift both AUROC and especially threshold-based Accuracy.

### Could you help?

1. Are the Table 10 numbers single-run or averaged over multiple runs? Any seed used?
2. **Would it be possible to share the ground-truth labels used in the paper — the `train_labels_no_reasoning.json` / `test_labels_no_reasoning.json` (or at least the `tool_necessary` lists) for all six models?** Since my env/data/model all match, comparing label sets directly would let us tell whether the gap comes from label randomness or from something else.
3. Could you also confirm the exact revision (commit hash) of `Qwen/Qwen3-4B-Instruct-2507` used for the paper?

Thank you very much for your time and for releasing the code!

---

## （可选）中文版

**标题**: [复现] Qwen3-4B-Instruct 多跳 probe 指标与论文不符（0.9257/0.8489 vs 0.966/0.947）——环境/数据/模型均一致，能否提供论文使用的 ground-truth 标签？

**正文**: 感谢开源！我未修改任何代码，按官方 pipeline 复现多跳 probe（no_reasoning + all_layers + reg=10000），结果 AUROC 0.9257 / Acc 0.8489，与论文 Table 10 的 0.966 / 0.947 差距明显。我已逐项排除以下因素：
1. 命令参数：与 `run_multihop_pipeline.sh` step 2/3 完全一致；
2. 依赖版本：torch 2.6.0+cu124 / transformers 4.55.2 / vllm 0.8.5 / accelerate 1.13.0 / scikit-learn 1.7.2，与 requirements.txt 逐项一致；
3. 数据：从 HF `cesun/When2Tool` 按 `utils._load_from_hf` 相同逻辑生成（single_hop 900/2250、multi_hop 180/450）；
4. 模型：`Qwen/Qwen3-4B-Instruct-2507`（ModelScope 下载，与 alias 解析一致）。

唯一剩余嫌疑：`tool_necessary` 标签来自 vLLM 的 hard_no_tool 推理，采样使用模型自带 generation_config（默认 temperature=0.7/top_p=0.8/top_k=20），代码中未固定 seed（vLLM 初始化与 SamplingParams 均无 seed），标签集逐次运行变化，Acc（阈值敏感）波动尤其大。

想请教：
1. Table 10 是单次还是多次平均？是否有固定 seed？
2. **能否分享论文使用的所有六个模型的 `train/test_labels_no_reasoning.json`（或至少 tool_necessary 列表）？** 我的环境/数据/模型均与你们一致，直接对比标签集即可判断差距是否来自标签随机性。
3. 能否确认论文所用 `Qwen/Qwen3-4B-Instruct-2507` 的确切 commit hash？

非常感谢！
