---
title: Run Classifier Reference
slug: /reference/run-classifier
description: Inference utilities for trained SAE-based classifiers.
---


## Module

`run_classifier.py`

## Classes

### `Normalizer`, `ZScoreNormalizer`, `L2Normalizer`

Normalization helpers used during inference.

### `ClassifierRunner`

High-level inference wrapper that loads:

- `classifier.pt`
- optional `config.json`
- optional normalization statistics

Useful methods:

- `predict_batch_raw`
- `predict_tensor`
- `predict_file`

Example:

```python
from pathlib import Path
from run_classifier import ClassifierRunner

runner = ClassifierRunner(Path("results/rq3/gemma-3-4b-it/classifier.pt"))
```

## Functions

- `resolve_activation_path`
- `apply_top_k`
- `apply_top_k_batch`
- `load_examples_index`
- `parse_args`
- `main`

## CLI examples

```bash
python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --activation assets/gemma-3-4b-it/sae_activations/benign_0-en-17.pt

python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --data-dir assets/gemma-3-4b-it \
  --example harmful_0:en benign_0:gu
```
