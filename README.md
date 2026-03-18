# Hidden in Plain Script

Research code for evaluating how LLM safety behavior changes when prompts are transliterated into non-Latin scripts. The pipeline pulls JailbreakBench behavior prompts, transliterates them into target scripts, runs a target model, and uses an LLM judge to label refusals and harmfulness. A logit-lens notebook is included for inspecting internal representations.

**Repository Layout**
- `process_dataset.py` downloads JailbreakBench JBB-Behaviors and then transliterates it into target scripts.
- `transliterate.py` transliterates sentences using the Google transliteration API or an LLM.
- `rq1.py` runs the main experiment and writes JSON results.
- `judge.py` defines the judge prompt and structured output schema.
- `utils.py` contains model generation and logit-lens tracing helpers.
- `logit_lens_transliteration.ipynb` explores logit-lens analysis (adapted from an NNSight tutorial).
- `assets/transliterations.json` caches transliterations.
- `results/` stores experiment outputs.

**Setup**
- Python 3.14 is required (see `pyproject.toml`).
- Install dependencies with Poetry:

```bash
poetry install
```

- Or use pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Configure environment variables:

```bash
cp .env.example .env
```

Fill in `HF_TOKEN` for Hugging Face model access. If you use LLM-based transliteration or judging, set the provider-specific API keys required by their SDKs.

**Dataset + Transliteration Pipeline**
`process_dataset.py` pulls the JailbreakBench JBB-Behaviors split (harmful and benign), saves it to JSON, and then transliterates each prompt into one or more target scripts.

```bash
python process_dataset.py -l gu te
```

This writes to `assets/transliterations.json` by default. Each entry includes `en`, `prompt_id`, `behavior`, `split`, plus one key per target language code.

**Transliterate Directly**
You can transliterate your own sentences or an existing JSON list.

```bash
# Sentences on the command line
python transliterate.py -s "Hello world" -l gu

# Transliterate a JSON list or list of objects with an `en` field
python transliterate.py -f assets/transliterations.json -l gu te

# LLM-based transliteration
python transliterate.py -s "Hello world" --method llm --provider openai --model <model>
```

**Run the Experiment**
`rq1.py` loads transliterated prompts, generates model responses with NNSight, and uses an LLM judge to label refusals and harmfulness. Results are saved in `results/` with a timestamped filename.

```bash
python rq1.py -d assets/transliterations.json -l gu --limit 50
```

**Results**
Output JSON includes the variant (language code), transliterated prompt, model response, and judge annotations (`judge_refused`, `judge_harmful`, `judge_reason`).
