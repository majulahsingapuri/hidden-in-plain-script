---
title: Sweep Reference
slug: /reference/run-experiments
description: Random-search hyperparameter sweeps for classifier training.
---


## Module

`run_experiments.py`

## Functions

### `parse_args()`

Parses sweep configuration such as `--trials`, `--metric`, and device overrides.

### `ensure_output_dir(path)`

Creates the sweep output directory, defaulting to `results/rq3/sweeps/<timestamp>/`.

### `log_uniform(rng, low, high)`

Samples learning-rate-like values in log space.

### `sample_config(rng)`

Builds one random hyperparameter proposal.

### `metric_from_metrics(metrics, metric_name)`

Extracts the optimization target from `metrics.json`.

### `build_cmd(args, output_root, model_name, config)`

Creates the `train_classifier.py` command for one trial.

### `main()`

Runs the sweep and writes `results.json` plus `best.json` as it goes.

## CLI example

```bash
python run_experiments.py \
  --data-dir assets/gemma-3-4b-it \
  --trials 50 \
  --metric val_f1
```
