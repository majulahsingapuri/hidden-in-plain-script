---
title: Config Reference
slug: /reference/config
description: Environment-backed runtime configuration.
---


## Module

`config.py`

## Class: `Config`

`Config` loads runtime settings from `.env` and the current environment.

Fields:

- `model`: default Hugging Face model name
- `hf_token`: Hugging Face access token
- `judge_provider`: `openai`, `anthropic`, or `ollama`
- `judge_model_name`: judge model identifier

## Example

```python
from config import Config

cfg = Config()
print(cfg.judge_provider)
```
