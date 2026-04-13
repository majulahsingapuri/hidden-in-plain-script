---
title: Process Dataset Reference
slug: /reference/process-dataset
description: Dataset download and transliteration pipeline entry point.
---


## Module

`process_dataset.py`

## Functions

### `download_dataset(path, limit=None)`

Downloads the JailbreakBench `JBB-Behaviors` dataset and writes harmful plus benign rows to JSON.

### `main()`

CLI entry point that:

1. Downloads the base dataset.
2. Writes it to the output path.
3. Calls `transliterate.main` to enrich the same file with language variants.

## CLI example

```bash
python process_dataset.py -l gu hi te ta --limit 100
```
