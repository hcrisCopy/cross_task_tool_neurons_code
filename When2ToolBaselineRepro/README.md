# When2Tool Baseline Reproduction

This folder is an isolated reproduction harness for the original When2Tool
probe baseline. It does not write into any existing method folder or existing
probe output directory.

All generated files are stored under:

```text
../cross_task_tool_neurons_data/when2tool_baseline_repro
```

## Purpose

The goal is to check whether the original When2Tool code can reproduce the
reported all-layer probe numbers on the same train/test tasks before comparing
new neuron-selection methods. This keeps the baseline separate from
`ProbePrefill`, `SafetyKernel_*`, `PreciseShield_*`, `ResidualDecisionAnchors`,
and `ToolKnowledgeNeurons`.

## Stage 1: Export Official JSON Tasks

Convert the cached HF parquet dataset into the four JSON filenames expected by
the upstream When2Tool scripts.

```bash
python When2ToolBaselineRepro/export_when2tool_json.py \
  --raw-dataset-dir ../cross_task_tool_neurons_data/datasets/raw_when2tool \
  --output-dir ../cross_task_tool_neurons_data/when2tool_baseline_repro/data
```

Outputs:

```text
../cross_task_tool_neurons_data/when2tool_baseline_repro/data/tasks_v1_train.json
../cross_task_tool_neurons_data/when2tool_baseline_repro/data/tasks_v1_test.json
../cross_task_tool_neurons_data/when2tool_baseline_repro/data/tasks_v1_multihop_train.json
../cross_task_tool_neurons_data/when2tool_baseline_repro/data/tasks_v1_multihop_test.json
../cross_task_tool_neurons_data/when2tool_baseline_repro/data/manifest.json
```

## Stage 2: Official Feature Extraction

Single-hop:

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage extract \
  --subset single_hop \
  --model-alias qwen3-4b-instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-new-tokens 2048 \
  --max-rounds 12
```

Multi-hop:

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage extract \
  --subset multi_hop \
  --model-alias qwen3-4b-instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-new-tokens 2048 \
  --max-rounds 12
```

Outputs:

```text
../cross_task_tool_neurons_data/when2tool_baseline_repro/probe_data/qwen3-4b-instruct
../cross_task_tool_neurons_data/when2tool_baseline_repro/probe_data/qwen3-4b-instruct_multihop
```

Each output directory contains official `train/test_no_tool_outputs.json`,
`train/test_labels_no_reasoning.json`, and `train/test_hidden_no_reasoning.pt`.

## Stage 3: Official Probe Training

Single-hop:

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage train \
  --subset single_hop \
  --model-alias qwen3-4b-instruct \
  --reg 10000
```

Multi-hop self-trained:

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage train \
  --subset multi_hop \
  --model-alias qwen3-4b-instruct \
  --reg 10000
```

Single-hop probe transferred to multi-hop:

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage transfer \
  --model-alias qwen3-4b-instruct
```

## One Command

Run all stages in order. Existing complete outputs are skipped early.

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage all \
  --subset both \
  --model-alias qwen3-4b-instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-new-tokens 2048 \
  --max-rounds 12 \
  --reg 10000
```

## Clear Wrong Isolated Outputs

Use `--clean` only for this reproduction root. The script refuses to remove
paths outside `../cross_task_tool_neurons_data/when2tool_baseline_repro`.

```bash
python When2ToolBaselineRepro/run_official_baseline.py \
  --stage all \
  --subset both \
  --model-alias qwen3-4b-instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-new-tokens 2048 \
  --max-rounds 12 \
  --reg 10000 \
  --clean
```

## Summarize Results

```bash
python When2ToolBaselineRepro/summarize_baseline.py \
  --model-alias qwen3-4b-instruct
```

The summary reports label distributions, single-hop all-layer probe metrics,
multi-hop self-trained probe metrics, and single-to-multi transfer metrics when
the files exist.
