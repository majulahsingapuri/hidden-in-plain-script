---
title: RQ2 Reference
slug: /reference/rq2
description: Layer-wise token tracing pipeline.
---


## Module

`rq2.py`

## Functions

### `run_experiment(data_path, model_name, layers_path, norm_path, langs, ...)`

Loads prompt variants, traces token predictions at every layer, and saves the decoded token strings plus token IDs.

### `main()`

CLI entry point for RQ2.

## CLI example

```bash
python rq2.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  --layers-path model.language_model.layers \
  --norm-path model.language_model.norm \
  -l gu hi te ta
```

## Output fields

- `variant`
- `prompt`
- `prompt_id`
- `words`
- `tokens`
