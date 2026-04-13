---
title: RQ3 Reference
slug: /reference/rq3
description: SAE feature extraction over transliterated prompts.
---


## Module

`rq3.py`

## Core functions

### `load_model(model_name="google/gemma-3-4b-it")`

Loads the traced `nnsight.LanguageModel`.

### `load_sae(release, sae_id, device="cuda")`

Loads a pretrained SAE from SAE Lens.

### `extract_hidden_states(model, prompt, target_layer, layers_path, norm_path)`

Traces a prompt and returns the normalized hidden states at the target layer.

### `decompose_features(sae, hidden_states)`

Encodes hidden states into sparse SAE activations.

### `sae_features(model, prompt_text, sae, target_layer, layers_path, norm_path)`

Returns one mean-pooled feature vector for the prompt.

### `run_batch(data_path, output_path, model_name, sae_release, sae_id, ...)`

Main batch driver. Iterates prompt variants, computes SAE features, saves `.pt` vectors, and records progress in `sae_features.json`.

## Support functions

- `load_progress`
- `save_progress`
- `build_done_sets`
- `save_activation`
- `main`

## CLI example

```bash
python rq3.py \
  -d assets/transliterations.json \
  -o assets/gemma-3-4b-it/sae_features.json \
  --model google/gemma-3-4b-it \
  --layer 17 \
  --sae-release gemma-scope-2-4b-it-res-all \
  --sae-id layer_17_width_262k_l0_small \
  -l gu hi te ta
```
