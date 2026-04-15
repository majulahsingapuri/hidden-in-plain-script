---
title: Getting Started
slug: /getting-started
description: Installation, environment setup, and first-run commands.
---

## Prerequisites

- Python `3.14`
- Access to Hugging Face Hub
- A configured `.env` file
- CUDA for `rq1.py`

## Installation

### Poetry

```bash
poetry install
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Start from the checked-in example:

```bash
cp .env.example .env
```

Current example values:

```bash
HF_TOKEN=""
ANTHROPIC_API_KEY=""
JUDGE_PROVIDER="anthropic"
JUDGE_MODEL_NAME="claude-haiku-4-5"
```

Common settings:

- `HF_TOKEN`: required by the Hugging Face-backed experiment scripts.
- `JUDGE_PROVIDER`: one of `anthropic`, `openai`, or `ollama`.
- `JUDGE_MODEL_NAME`: model identifier for the configured judge provider.
- `OPENAI_API_KEY`: required when using OpenAI-backed judging or LLM transliteration.

## First dataset build

```bash
python process_dataset.py -l gu hi te ta
```

This creates `assets/transliterations.json`.

## First experiment run

```bash
python rq1.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  -l gu hi te ta
```

## Shortcut scripts

If you want the checked-in experiment defaults, use the shell wrappers:

```bash
./rq1.sh
./rq2.sh
./rq3.sh
```

## Notebooks

The repository also includes analysis notebooks:

- `logit_lens_transliteration.ipynb`
- `rq1_simple.ipynb`
- `rq1_analysis_deep.ipynb`
- `rq2.ipynb`

## Suggested reading order

1. [Data Model](./data-model.md)
2. [Workflow Overview](./workflows/overview.md)
3. [Classifier Workflow](./workflows/classifier.md)
