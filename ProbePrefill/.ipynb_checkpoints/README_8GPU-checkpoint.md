# ProbePrefill：单卡与单机八卡运行

PP-1/PP-2 不加载大模型：PP-1 从既有 activation 选取 CTD 特征，PP-2 用 CPU sklearn 训练线性 probe；两步保持单进程。PP-3、PP-4、PP-5 使用脚本内置的数据并行。

## 固定实验口径

`prompt_mode=current`、`reasoning_mode=no_reasoning`、`enable_thinking=false`、`max_rounds=10`、`max_new_tokens=2048`、`max_model_len=32768`、`record_mode=lite`、`seed=2026`。正式评测：`temperature=2.0`、`thresholds=0.1,0.3,0.5,0.7,0.9`、`n_runs=3`。

## PP-1 / PP-2

```bash
python ProbePrefill/pp_build_probe_features.py --model-alias qwen3-4b-instruct --subset all --max-train-samples 0 --max-test-samples 0 --sample-strategy balanced --require-per-type-labels --seed 2026
python ProbePrefill/pp_train_probe.py --model-alias qwen3-4b-instruct --subset all --reg 10000 --max-iter 2000 --threshold 0.5
```

## 单卡

```bash
python ProbePrefill/pp_eval_probe_prefill.py --model-alias qwen3-4b-instruct --subset all --gpus 0 --thresholds 0.1,0.3,0.5,0.7,0.9 --temperature 2.0 --n-runs 3 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --record-mode lite --seed 2026
```

## 八卡

```bash
python ProbePrefill/pp_eval_probe_prefill.py --model-alias qwen3-4b-instruct --subset all --gpus 0,1,2,3,4,5,6,7 --thresholds 0.1,0.3,0.5,0.7,0.9 --temperature 2.0 --n-runs 3 --max-rounds 10 --max-new-tokens 2048 --max-model-len 32768 --record-mode lite --seed 2026
```

PP-4 和 PP-5 使用相同的 `--gpus` 参数。八卡模式按稳定的 `rank::8` 拆分 test id；每卡一个独立模型，终端显示跨 worker 的全局完成题目数，结束后自动合并并重算 summary、delta 和因果验证指标。
