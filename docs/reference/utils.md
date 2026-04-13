---
title: Utility Reference
slug: /reference/utils
description: Shared helpers for tracing, batching, resource monitoring, and token script analysis.
---

## Module

`utils.py`

## Resource monitoring

### `ResourceMonitor`

Collects CPU, RAM, GPU, and VRAM utilization for long-running loops.

Important methods:

- `sample()`
- `tqdm_postfix()`
- `close()`

## Tracing and batching helpers

### `resolve_attr_path(root, path)`

Resolves dotted attribute paths with optional list indexing.

### `generate_trace(model, prompt_texts, layers_path, norm_path, return_max_probs=False)`

Runs prompt tracing and returns top token IDs plus decoded tokens for each layer.

### `build_variants(langs)`

Builds the variant list:

- `en`
- `<lang>`
- `<lang>_en`
- `en_<lang>`

### `iter_work_items(prompts_list, variants_list, completed=None)`

Yields work items for every prompt and variant pair, skipping completed pairs when provided.

### `iter_batches(items_iter, size)`

Groups an iterator into fixed-size lists.

### `count_work_items(prompts_list, variants_list, completed=None)`

Counts pending prompt-variant pairs.

### `count_batches(total_items, batch_size)`

Computes the number of batches needed.

## Token script helpers

These functions support script analysis for decoded tokens:

- `_in_ranges`
- `_char_script`
- `_normalize_token_text`
- `_token_text_to_script`
- `token_id_to_script`
- `build_token_script_map`

## Example

```python
from utils import build_variants

variants = build_variants(["gu", "hi"])
print(variants)
```
