---
title: Train Classifier Reference
slug: /reference/train-classifier
description: Linear-probe training pipeline over SAE activations.
---


## Module

`train_classifier.py`

## Data structures

### `Example`

Metadata for one activation example.

### `ActivationDataset`

PyTorch dataset that loads and preprocesses activation vectors on demand.

### `Normalizer`, `ZScoreNormalizer`, `L2Normalizer`

Normalization helpers applied before training and evaluation.

## Core functions

### `load_examples(data_dir, target_layer=None, max_items=None)`

Loads metadata rows from `sae_features.json` and assigns binary labels from `prompt_id`.

### `stratified_split(examples, seed)`

Creates reproducible train, validation, and test splits.

### `compute_zscore_stats(dataset, batch_size, num_workers)`

Computes train-set feature statistics for z-score normalization.

### `train_epoch(model, loader, device, optimizer, ...)`

Runs one training epoch and returns average loss.

### `fit_temperature(y_true, logits)`

Fits a scalar temperature on validation logits.

### `tune_threshold(y_true, y_prob)`

Searches thresholds in `0.00` through `1.00` to maximize validation F1.

### `main()`

CLI entry point that trains, evaluates, calibrates, and writes model artifacts.

## CLI example

```bash
python train_classifier.py \
  --data-dir assets/gemma-3-4b-it \
  --output-root results/rq3 \
  --model-name gemma-3-4b-it \
  --epochs 20 \
  --batch-size 128 \
  --top-k 4096
```
