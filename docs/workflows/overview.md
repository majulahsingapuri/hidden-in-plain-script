---
title: Workflow Overview
slug: /workflows/overview
description: End-to-end workflow from raw prompts to judged outputs and SAE features.
---

## Step 1: Build the transliterated dataset

```bash
python process_dataset.py -l gu hi te ta
```

This downloads JailbreakBench, writes base rows, and then enriches them with translations and transliterations.

## Step 2: Run judged generations with RQ1

```bash
python rq1.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  -l gu hi te ta \
  --batch-size 16 \
  --judge-concurrency 4
```

Use `--output` and `--resume` when you want resumable runs.

## Step 3: Trace internal token predictions with RQ2

```bash
python rq2.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  --layers-path model.language_model.layers \
  --norm-path model.language_model.norm \
  -l gu hi te ta
```

## Step 4: Extract SAE features with RQ3

```bash
mkdir -p assets/gemma-3-4b-it

python rq3.py \
  -d assets/transliterations.json \
  -o assets/gemma-3-4b-it/sae_features.json \
  --model google/gemma-3-4b-it \
  --layer 17 \
  --sae-release gemma-scope-2-4b-it-res-all \
  --sae-id layer_17_width_262k_l0_small \
  -l gu hi te ta
```

## Step 5: Train and serve a classifier

Continue with the dedicated [Classifier Workflow](./classifier.md).
