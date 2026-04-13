---
title: Classifier Workflow
slug: /workflows/classifier
description: Training, evaluating, and sweeping the SAE-based linear probe.
---

## Inputs

The classifier training pipeline expects:

- `assets/<model-name>/sae_features.json`
- `assets/<model-name>/sae_activations/*.pt`

These are produced by `rq3.py`.

## Train a probe

```bash
python train_classifier.py \
  --data-dir assets/gemma-3-4b-it \
  --output-root results/rq3 \
  --model-name gemma-3-4b-it
```

Useful options:

- `--normalize {none,zscore,l2}`
- `--top-k 4096`
- `--batch-size 128`
- `--epochs 20`
- `--device cuda`
- `--threshold 0.5`

## Inspect outputs

Training writes:

- `classifier.pt`: learned linear layer and inference metadata
- `config.json`: run configuration and derived dimensions
- `metrics.json`: train, validation, and test metrics
- `splits.json`: dataset splits and example metadata
- `normalizer.pt`: z-score statistics when enabled

## Run inference

### Score a saved activation file

```bash
python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --activation assets/gemma-3-4b-it/sae_activations/benign_0-en-17.pt
```

### Score indexed examples

```bash
python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --data-dir assets/gemma-3-4b-it \
  --example harmful_0:en benign_0:gu
```

### Save JSONL output

```bash
python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --data-dir assets/gemma-3-4b-it \
  --example harmful_0:en \
  --out predictions.jsonl
```

## Sweep hyperparameters

```bash
python run_experiments.py \
  --data-dir assets/gemma-3-4b-it \
  --trials 50
```

Sweep results are written to `results/rq3/sweeps/<timestamp>/`.
