---
title: Judge Reference
slug: /reference/judge
description: LLM judge schema and helper functions.
---


## Module

`judge.py`

## Constants

### `JUDGE_SYSTEM_PROMPT`

The system prompt used to instruct the judge model.

## Class: `EvaluationResponse`

Structured judge output with:

- `refused`
- `harmful`
- `gibberish`

## Functions

### `_is_rate_limit_error(exc)`

Detects whether an exception likely came from API rate limiting.

### `_retry_sleep(attempt, base_delay=1.0, max_delay=30.0)`

Sleeps with jittered exponential backoff.

### `judge_response(agent, prompt_text, response_text)`

Synchronously classifies one prompt-response pair.

Example:

```python
from pydantic_ai import Agent
from judge import EvaluationResponse, JUDGE_SYSTEM_PROMPT, judge_response

judge = Agent(
    "openai:gpt-4.1-mini",
    system_prompt=JUDGE_SYSTEM_PROMPT,
    output_type=EvaluationResponse,
)

verdict = judge_response(judge, "Prompt text", "Model response")
print(verdict.refused, verdict.harmful, verdict.gibberish)
```

### `judge_response_async(agent, prompt_text, response_text)`

Async equivalent used by `rq1.py` for concurrent judging.
