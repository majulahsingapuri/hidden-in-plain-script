---
title: RQ1 Reference
slug: /reference/rq1
description: Generation and judging pipeline over transliterated prompts.
---


## Module

`rq1.py`

## Functions

### `judge_batch(judge, batch_items, responses, concurrency)`

Runs asynchronous judging with a concurrency limit.

### `run_experiment(data_path, model_name, langs, ...)`

Runs the full RQ1 pipeline:

1. Loads transliterated prompts.
2. Expands language variants.
3. Generates responses with Gemma.
4. Judges each response for refusal, harmfulness, and gibberish.
5. Saves JSON results incrementally.

### `main()`

CLI entry point for RQ1.

## CLI example

```bash
python rq1.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  -l gu hi te ta \
  --batch-size 16 \
  --judge-concurrency 4
```

## Output fields

- `variant`
- `prompt`
- `prompt_id`
- `response`
- `refused`
- `harmful`
- `gibberish`
