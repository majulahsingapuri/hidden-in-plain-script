---
title: Transliteration Reference
slug: /reference/transliterate
description: Transliteration helpers and CLI entry points.
---


## Module

`transliterate.py`

## Main data model

### `Transliterations`

Structured LLM output that stores transliterated words in the `words` field.

## Core functions

### `transliterate_sentence(sentence, lang_code, method)`

Transliterates one English sentence into the target script.

Example:

```python
from transliterate import transliterate_sentence

text = transliterate_sentence("Hello world", "gu", "transliterator")
print(text)
```

### `process_sentences(sentences, path, lang_codes, method)`

Main programmatic API. Reuses cached rows, transliterates missing ones, and adds:

- `en_<lang>`
- `<lang>`
- `<lang>_en`

Example:

```python
from pathlib import Path
from transliterate import process_sentences

rows = process_sentences(
    ["Hello world"],
    Path("assets/transliterations.json"),
    ["gu", "hi"],
    "transliterator",
)
```

## Important helpers

- `_translate_many_googletrans_async`
- `_translate_many_googletrans`
- `_transliterate_with_retry`
- `_tokenize_with_separators`
- `_build_input_tools_url`
- `_estimate_url_len`
- `_chunk_words_for_url`
- `_fetch_batch_transliterations`
- `_transliterate_sentence_batched`
- `_transliterate_sentence_per_word`
- `load_existing`
- `save_to_json`
- `main`

## CLI example

```bash
python transliterate.py -s "Hello world" -l gu hi te ta
python transliterate.py -f assets/transliterations.json --method llm --provider openai --model gpt-4.1-mini -l gu
```
