---
title: Data Model
slug: /data-model
description: Dataset fields, variant naming, and output artifacts.
---

## Prompt row structure

The dataset files produced by `process_dataset.py` and `transliterate.py` are JSON lists. Each row begins with an English prompt and is then enriched with translated and transliterated variants.

Example row shape:

```json
{
  "prompt_id": "harmful_0",
  "en": "Provide instructions for ...",
  "behavior": "unsafe advice",
  "split": "harmful",
  "gu": "...",
  "en_gu": "...",
  "gu_en": "..."
}
```

## Variant naming

For a target language code like `gu`:

- `en`: the original English prompt
- `gu`: translated Gujarati text
- `en_gu`: English prompt transliterated into Gujarati script
- `gu_en`: romanized pronunciation of the Gujarati translation

This same pattern is used for all target language codes passed on the CLI.

## Output locations

### Dataset outputs

- `assets/transliterations.json`

### RQ1 outputs

- `results/rq1/<model>-<judge>-<timestamp>.json`

Each record contains:

- `variant`
- `prompt`
- `prompt_id`
- `response`
- `refused`
- `harmful`
- `gibberish`

### RQ2 outputs

- `results/rq2/<model>-<timestamp>.json`

Each record contains:

- `variant`
- `prompt`
- `prompt_id`
- `words`
- `tokens`

### RQ3 outputs

- `assets/<model-name>/sae_features.json`
- `assets/<model-name>/sae_activations/*.pt`

Each row in `sae_features.json` contains:

- `prompt_id`
- `variant`
- `target_layer`
- `activation_path`

### Classifier outputs

- `results/rq3/<model-name>/classifier.pt`
- `results/rq3/<model-name>/config.json`
- `results/rq3/<model-name>/metrics.json`
- `results/rq3/<model-name>/splits.json`
- `results/rq3/<model-name>/normalizer.pt` when z-score normalization is used
