---
title: Intro
slug: /intro
description: Overview of the Hidden in Plain Script research codebase.
---

Hidden in Plain Script is a research codebase for studying how model safety behavior changes when English prompts are transliterated into non-Latin scripts.

The repository supports four main tasks:

1. Build transliterated prompt datasets from JailbreakBench.
2. Run model generation and judge responses for refusal, harmfulness, and gibberish.
3. Trace internal token predictions and extract SAE features.
4. Train and serve a linear probe over saved SAE activations.

## Repository map

- `process_dataset.py`: download JailbreakBench and build the transliterated dataset.
- `transliterate.py`: transliterate raw text or JSON datasets.
- `rq1.py`: run judged generations over prompt variants.
- `rq2.py`: trace layer-wise token predictions.
- `rq3.py`: extract mean-pooled SAE features.
- `train_classifier.py`: train a linear probe.
- `run_classifier.py`: run inference with a trained probe.
- `run_experiments.py`: sweep classifier hyperparameters.
- `rq1.sh`, `rq2.sh`, `rq3.sh`: shell wrappers for the main experiment runs.
- notebooks: exploratory analysis and logit-lens inspection.
- `assets/`: cached transliterations and activation artifacts.
- `results/`: experiment outputs and trained classifier artifacts.

## Documentation map

- [Getting Started](./getting-started.md)
- [Data Model](./data-model.md)
- [Workflow Overview](./workflows/overview.md)
- [Classifier Workflow](./workflows/classifier.md)
- [Config Reference](./reference/config.md)
- [Process Dataset Reference](./reference/process-dataset.md)
- [Transliteration Reference](./reference/transliterate.md)
- [RQ1 Reference](./reference/rq1.md)
- [RQ2 Reference](./reference/rq2.md)
- [RQ3 Reference](./reference/rq3.md)
- [Training Reference](./reference/train-classifier.md)
- [Inference Reference](./reference/run-classifier.md)
- [Sweep Reference](./reference/run-experiments.md)
- [Judge Reference](./reference/judge.md)
- [Utility Reference](./reference/utils.md)

## Quick start

```bash
poetry install
cp .env.example .env
python process_dataset.py -l gu hi te ta
python rq1.py -d assets/transliterations.json --model google/gemma-3-4b-it -l gu hi te ta
```
