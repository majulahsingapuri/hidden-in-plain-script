# Hidden in Plain Script

Research code for studying how LLM safety behavior changes when English prompts are transliterated into non-Latin scripts. The repo builds transliterated variants of JailbreakBench prompts, runs model evaluations, traces internal token predictions, extracts SAE features, and trains a linear probe over those features.

## What Is In This Repo

- `process_dataset.py`: downloads `JailbreakBench/JBB-Behaviors`, creates harmful/benign prompt rows, and transliterates them.
- `transliterate.py`: transliterates free-form sentences or JSON datasets and caches the results.
- `rq1.py`: runs response generation plus LLM judging for refusal, harmfulness, and gibberish.
- `rq2.py`: traces the model layer by layer and saves top predicted token IDs and decoded tokens.
- `rq3.py`: extracts mean-pooled SAE activations for each prompt variant.
- `train_classifier.py`: trains a linear probe on saved SAE activations.
- `run_classifier.py`: loads a trained probe and scores activation files or indexed examples.
- `run_experiments.py`: random-search sweep for `train_classifier.py`.
- `rq1.sh`, `rq2.sh`, `rq3.sh`: shell wrappers around the main experiment CLIs.
- `docs/`: Docusaurus-friendly Markdown documentation for the codebase and workflows.
- `assets/`: transliteration caches and SAE activation artifacts.
- `results/`: experiment outputs, trained probes, metrics, and splits.

## Documentation

If you are starting from scratch, read these in order:

1. `docs/intro.md`
2. `docs/getting-started.md`
3. `docs/data-model.md`
4. `docs/workflows/overview.md`
5. `docs/workflows/classifier.md`

The `docs/` tree is formatted for Docusaurus and mirrors the main workflows plus per-module references.

## Setup

Python `3.14` is required by `pyproject.toml`.

With Poetry:

```bash
poetry install
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local env file:

```bash
cp .env.example .env
```

The tracked example contains:

```bash
HF_TOKEN=""
ANTHROPIC_API_KEY=""
JUDGE_PROVIDER="anthropic"
JUDGE_MODEL_NAME="claude-haiku-4-5"
```

Notes:

- `HF_TOKEN` is required for Hugging Face model access.
- `JUDGE_PROVIDER` supports `anthropic`, `openai`, and `ollama`.
- If you use OpenAI-backed judging or LLM transliteration, you will also need the provider credentials expected by that SDK, for example `OPENAI_API_KEY`.
- `rq1.py` explicitly requires CUDA for batched generation.

## Data Format

`process_dataset.py` and `transliterate.py` write a JSON list where each item starts with English text under `en` and then adds variant fields for each language:

- `<lang>`: translated text
- `en_<lang>`: transliterated English rendered in the target script
- `<lang>_en`: romanized pronunciation of the translated text

For JailbreakBench rows, the file also includes `prompt_id`, `behavior`, and `split`.

## End-to-End Workflow

### 1. Build The Transliterated Dataset

```bash
python process_dataset.py -l gu hi te ta
```

By default this writes `assets/transliterations.json`.

You can transliterate ad hoc text or an existing JSON file directly:

```bash
python transliterate.py -s "Hello world" -l gu hi te ta
python transliterate.py -f assets/transliterations.json -l gu hi te ta
python transliterate.py -s "Hello world" --method llm --provider openai --model gpt-4.1-mini -l gu
```

### 2. Run RQ1: Safety Behavior Under Transliteration

```bash
python rq1.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  -l gu hi te ta \
  --batch-size 16 \
  --judge-concurrency 4
```

This writes a timestamped JSON file under `results/rq1/`. Each record includes:

- `variant`
- `prompt`
- `prompt_id`
- `response`
- `refused`
- `harmful`
- `gibberish`

To resume into a specific file:

```bash
python rq1.py \
  -d assets/transliterations.json \
  --output results/rq1/gemma-run.json \
  --resume
```

### 3. Run RQ2: Layer-Wise Token Trace

```bash
python rq2.py \
  -d assets/transliterations.json \
  --model google/gemma-3-4b-it \
  --layers-path model.language_model.layers \
  --norm-path model.language_model.norm \
  -l gu hi te ta
```

This writes a timestamped JSON file under `results/rq2/`. Each row stores the input variant, prompt ID, decoded `words`, and token IDs traced at every layer.

If you trace a different architecture, override `--layers-path` and `--norm-path` with the correct dotted paths inside the `nnsight.LanguageModel` wrapper.

### 4. Run RQ3: SAE Feature Extraction

For model-specific SAE outputs, create a per-model asset directory first:

```bash
mkdir -p assets/gemma-3-4b-it
```

Then extract features:

```bash
python rq3.py \
  -d assets/transliterations.json \
  -o assets/gemma-3-4b-it/sae_features.json \
  --model google/gemma-3-4b-it \
  --layer 17 \
  --sae-release gemma-scope-2-4b-it-res-all \
  --sae-id layer_17_width_262k_l0_small \
  --layers-path model.language_model.layers \
  --norm-path model.language_model.norm \
  -l gu hi te ta
```

This produces:

- `assets/gemma-3-4b-it/sae_features.json`
- `assets/gemma-3-4b-it/sae_activations/*.pt`

Each `sae_features.json` row records `prompt_id`, `variant`, `target_layer`, and `activation_path`.

### 5. Train The Linear Probe

```bash
python train_classifier.py \
  --data-dir assets/gemma-3-4b-it \
  --output-root results/rq3 \
  --model-name gemma-3-4b-it
```

The training output directory contains:

- `classifier.pt`
- `config.json`
- `metrics.json`
- `splits.json`
- `normalizer.pt` when z-score normalization is used

Example trained artifacts already present in the repo live under `results/rq3/gemma-3-4b-it`.

### 6. Run Inference With A Trained Probe

Score a saved activation file:

```bash
python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --activation assets/gemma-3-4b-it/sae_activations/benign_0-en-17.pt
```

Or look up examples by `prompt_id:variant` from `sae_features.json`:

```bash
python run_classifier.py \
  --model results/rq3/gemma-3-4b-it \
  --data-dir assets/gemma-3-4b-it \
  --example harmful_0:en benign_0:gu
```

Pass `--out predictions.jsonl` to write JSONL instead of printing to stdout.

### 7. Sweep Probe Hyperparameters

```bash
python run_experiments.py \
  --data-dir assets/gemma-3-4b-it \
  --trials 50
```

Sweep outputs are written under `results/rq3/sweeps/<timestamp>/` with `results.json` and `best.json`.

## Shell Shortcuts

The repo includes wrappers with the same core settings used in the paper workflow:

```bash
./rq1.sh
./rq2.sh
./rq3.sh
```

Use these when you want the checked-in defaults instead of spelling out the full Python CLI flags yourself.

## Results Layout

- `results/rq1/`: model generations plus judge labels
- `results/rq2/`: layer-wise token traces
- `results/rq3/`: trained classifiers, metrics, splits, and sweeps

## Notebooks

The repository also includes exploratory notebooks:

- `logit_lens_transliteration.ipynb`
- `rq1_simple.ipynb`
- `rq1_analysis_deep.ipynb`
- `rq2.ipynb`
