import json
import argparse
from pathlib import Path
from typing import Literal, Optional, Union
from google.transliteration import transliterate_word

# Set by CLI when using --method llm
LLM_PROVIDER = None
LLM_MODEL = None


def _llm_transliterate(prompt: str) -> str:
    """Call the chosen LLM provider and return the generated text."""

    if LLM_PROVIDER == "openai":
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Install it with 'pip install openai' to use --provider openai."
            ) from e

        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    if LLM_PROVIDER == "anthropic":
        try:
            from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed. Install it with 'pip install anthropic' to use --provider anthropic."
            ) from e

        client = Anthropic()
        resp = client.completions.create(
            model=LLM_MODEL,
            prompt=f"{HUMAN_PROMPT}{prompt}{AI_PROMPT}",
            max_tokens_to_sample=1024,
        )
        return resp.completion.strip()

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def transliterate_sentence(
    sentence: str, lang_code: str, method: Literal["transliterator", "llm"]
) -> str:
    """Transliterate each word in a sentence to the target language."""
    if method == "transliterator":
        return " ".join(
            [
                (result[0] if (result := transliterate_word(word, lang_code)) else word)
                for word in sentence.split()
            ]
        )
    elif method == "llm":
        # Prompt used to ask the LLM for a transliteration.
        prompt = (
            "Transliterate the following English sentence into {lang_code} script. "
            "Keep punctuation and spacing exactly as in the input, and output only the transliteration.\n"
            "Sentence: {sentence}\n"
        ).format(lang_code=lang_code, sentence=sentence)

        return _llm_transliterate(prompt)
    else:
        raise ValueError("Invalid transliteration method")


def load_existing(path: Path) -> list[dict]:
    """Load transliterations from an existing JSON file."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_to_json(data: list[dict], path: Path):
    """Write the full data list to the JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_sentences(
    sentences: Union[list[str], list[dict]],
    path: Optional[Path],
    lang_codes: list[str],
    method: Literal["transliterator", "llm"],
) -> list[dict]:
    """
    For each sentence:
      - If already in the JSON cache, reuse it (no API call).
      - If missing, transliterate it and append to the cache.
    Saves the updated JSON at the end if anything new was added.
    """

    cache = {}
    if path:
        existing = load_existing(path)
        # Build a lookup dict keyed by the English sentence
        cache = {entry["en"]: entry for entry in existing}

    results = []
    new_count = 0

    for sentence in sentences:
        data = None
        if isinstance(sentence, dict):
            data = sentence.copy()
            sentence = sentence["en"]
        if cache and sentence in cache and set(lang_codes) <= cache[sentence].keys():
            print(f"[CACHE]  {sentence}")
            results.append(cache[sentence])
        else:
            print(f"[NEW]    {sentence}")
            entry = {
                "en": sentence,
            } | {
                lang_code: transliterate_sentence(sentence, lang_code, method)
                for lang_code in lang_codes
            }
            if data:
                entry |= data
            results.append(entry)
            cache[sentence] = entry
            new_count += 1

    # Save only if there were new transliterations
    if path:
        if new_count > 0:
            all_entries = list(cache.values())
            save_to_json(all_entries, path)
            print(
                f"\nAdded {new_count} new entries. Total: {len(all_entries)} in {path}"
            )
        else:
            print(f"\nAll {len(results)} sentences found in cache.")
    else:
        print(f"\nProcessed {new_count} sentences")

    return results


def main(args=None):
    DEFAULT_SENTENCES = ["Yo what is up developer?", "How do you commit a hate crime?"]

    parser = argparse.ArgumentParser(
        description="Transliterate sentences into one or more target languages."
    )
    parser.add_argument(
        "--sentences",
        "-s",
        nargs="+",
        help="Sentences to transliterate. If omitted, defaults will be used unless --sentences-file is provided.",
    )
    parser.add_argument(
        "--sentences-file",
        "-f",
        type=Path,
        help="Path to a JSON file containing a list of sentences (strings) or list of objects with key `en` containing the english sentence",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default=Path.cwd() / "assets" / "transliterations.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--langs",
        "-l",
        nargs="+",
        default=["gu"],
        help="Target language codes (e.g., gu te).",
    )
    parser.add_argument(
        "--method",
        choices=["transliterator", "llm"],
        default="transliterator",
        help="Transliteration method to use.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="LLM provider to use when --method=llm.",
    )
    parser.add_argument(
        "--model",
        help="LLM model name to use when --method=llm.",
    )

    if not args:
        args = parser.parse_args()

    if args.method == "llm":
        if not args.provider or not args.model:
            parser.error("--provider and --model are required when --method=llm")
        LLM_PROVIDER = args.provider
        LLM_MODEL = args.model

    if args.sentences_file:
        with open(args.sentences_file, "r", encoding="utf-8") as f:
            sentences = json.load(f)
        if not isinstance(sentences, list) or not all(
            isinstance(s, str) or isinstance(s, dict) for s in sentences
        ):
            raise ValueError("sentences-file must be a JSON list of strings/dicts")
    elif args.sentences:
        sentences = args.sentences
    else:
        sentences = DEFAULT_SENTENCES

    return process_sentences(sentences, args.output_path, args.langs, args.method)


if __name__ == "__main__":
    main()
