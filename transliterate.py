import json
import argparse
import asyncio
import time
import re
from urllib.parse import quote_plus
from pathlib import Path
from typing import Literal, Optional, Union
from google.transliteration import transliterate_word
from googletrans import Translator
from pydantic_ai import Agent
from unidecode import unidecode
import requests
from tqdm import tqdm

# Set by CLI when using --method llm
LLM_PROVIDER = None
LLM_MODEL = None

G_API_DEFAULT = "https://inputtools.google.com/request?text=%s&itc=%s-t-i0&num=%d"
G_API_CHINESE = "https://inputtools.google.com/request?text=%s&itc=%s-t-i0-%s&num=%d"
CHINESE_LANGS = {"yue-hant", "zh", "zh-hant"}
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_WORD_CACHE: dict[tuple[str, str], str] = {}


async def _translate_many_googletrans_async(
    sentences: list[str], lang_codes: list[str]
) -> dict[str, list]:
    translator = Translator()
    try:
        results: dict[str, list] = {}
        for lang_code in lang_codes:
            if not sentences:
                results[lang_code] = []
                continue
            translated = translator.translate(sentences, dest=lang_code, src="en")
            if asyncio.iscoroutine(translated):
                translated = await translated
            if not isinstance(translated, list):
                translated = [translated]
            results[lang_code] = translated
        return results
    finally:
        client = getattr(translator, "client", None)
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()


def _translate_many_googletrans(
    sentences: list[str], lang_codes: list[str]
) -> dict[str, list]:
    return asyncio.run(_translate_many_googletrans_async(sentences, lang_codes))


def _transliterate_with_retry(
    sentence: str, lang_code: str, attempts: int = 3, backoff_s: float = 0.5
) -> str:
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            return _transliterate_sentence_batched(sentence, lang_code)
        except requests.exceptions.RequestException as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(backoff_s * (2**i))
        except ValueError as e:
            last_err = e
            # Fallback to per-word requests if batch response is invalid.
            try:
                return _transliterate_sentence_per_word(sentence, lang_code)
            except (requests.exceptions.RequestException, ValueError) as e2:
                last_err = e2
                if i < attempts - 1:
                    time.sleep(backoff_s * (2**i))
    print(
        f"[WARN] Transliteration failed for lang={lang_code}; using original text. Error: {last_err}"
    )
    return sentence


def _tokenize_with_separators(sentence: str) -> list[tuple[str, bool]]:
    """Return list of (segment, is_word) preserving separators."""
    parts: list[tuple[str, bool]] = []
    pos = 0
    for match in _WORD_RE.finditer(sentence):
        if match.start() > pos:
            parts.append((sentence[pos : match.start()], False))
        parts.append((match.group(0), True))
        pos = match.end()
    if pos < len(sentence):
        parts.append((sentence[pos:], False))
    return parts


def _build_input_tools_url(
    text: str, lang_code: str, max_suggestions: int = 1, input_scheme: str = "pinyin"
) -> str:
    if lang_code in CHINESE_LANGS:
        return G_API_CHINESE % (
            quote_plus(text),
            lang_code,
            input_scheme,
            max_suggestions,
        )
    return G_API_DEFAULT % (quote_plus(text), lang_code, max_suggestions)


def _estimate_url_len(
    text: str, lang_code: str, max_suggestions: int = 1, input_scheme: str = "pinyin"
) -> int:
    return len(_build_input_tools_url(text, lang_code, max_suggestions, input_scheme))


def _chunk_words_for_url(
    words: list[str], lang_code: str, max_url_len: int = 1800
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for word in words:
        candidate = current + [word]
        text = " ".join(candidate)
        if _estimate_url_len(text, lang_code) <= max_url_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = [word]
    if current:
        chunks.append(current)
    return chunks


def _fetch_batch_transliterations(
    words: list[str], lang_code: str, max_suggestions: int = 1
) -> Optional[dict[str, str]]:
    if not words:
        return {}
    text = " ".join(words)
    url = _build_input_tools_url(text, lang_code, max_suggestions)
    response = requests.get(url, allow_redirects=False, timeout=5)
    payload = response.json()
    if response.status_code != 200 or not payload or payload[0] != "SUCCESS":
        return None
    entries = payload[1]
    if not isinstance(entries, list) or len(entries) != len(words):
        return None
    results: dict[str, str] = {}
    for i, entry in enumerate(entries):
        try:
            candidates = entry[1]
        except Exception:
            return None
        if candidates:
            results[words[i]] = candidates[0]
        else:
            results[words[i]] = words[i]
    return results


def _transliterate_sentence_batched(sentence: str, lang_code: str) -> str:
    parts = _tokenize_with_separators(sentence)
    word_tokens = [segment for segment, is_word in parts if is_word]
    if not word_tokens:
        return sentence

    missing_words: list[str] = []
    seen: set[str] = set()
    for word in word_tokens:
        key = (lang_code, word.lower())
        if key not in _WORD_CACHE and word.lower() not in seen:
            seen.add(word.lower())
            missing_words.append(word.lower())

    for chunk in _chunk_words_for_url(missing_words, lang_code):
        batch = _fetch_batch_transliterations(chunk, lang_code, max_suggestions=1)
        if batch is None:
            raise ValueError("Batch response mismatch")
        for word_lower, translit in batch.items():
            _WORD_CACHE[(lang_code, word_lower)] = translit

    rebuilt: list[str] = []
    for segment, is_word in parts:
        if not is_word:
            rebuilt.append(segment)
            continue
        key = (lang_code, segment.lower())
        rebuilt.append(_WORD_CACHE.get(key, segment))
    return "".join(rebuilt)


def _transliterate_sentence_per_word(sentence: str, lang_code: str) -> str:
    parts = _tokenize_with_separators(sentence)
    if not any(is_word for _, is_word in parts):
        return sentence
    rebuilt: list[str] = []
    for segment, is_word in parts:
        if not is_word:
            rebuilt.append(segment)
            continue
        key = (lang_code, segment.lower())
        cached = _WORD_CACHE.get(key)
        if cached is not None:
            rebuilt.append(cached)
            continue
        result = transliterate_word(segment, lang_code)
        translit = result[0] if result else segment
        _WORD_CACHE[key] = translit
        rebuilt.append(translit)
    return "".join(rebuilt)


def transliterate_sentence(
    sentence: str, lang_code: str, method: Literal["transliterator", "llm"]
) -> str:
    """Transliterate each word in a sentence to the target language."""
    if method == "transliterator":
        return _transliterate_with_retry(sentence, lang_code)
    elif method == "llm":
        # Prompt used to ask the LLM for a transliteration.
        system_prompt = (
            "Transliterate the following sentence into {lang_code} script. "
            "Keep punctuation and spacing exactly as in the input, and output only the transliteration."
        ).format(lang_code=lang_code)
        translator_ai = Agent(
            f"{LLM_PROVIDER}:{LLM_MODEL}", system_prompt=system_prompt
        )
        return translator_ai.run_sync(f"Sentence:{sentence}").output
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
    pending_sentences: list[str] = []
    pending_entries: list[dict] = []

    for sentence in tqdm(sentences, position=0):
        data = None
        if isinstance(sentence, dict):
            data = sentence.copy()
            sentence = sentence["en"]
        if cache and sentence in cache and set(lang_codes) <= cache[sentence].keys():
            results.append(cache[sentence])
        else:
            entry = {
                "en": sentence,
            } | {
                f"en_{lang_code}": transliterate_sentence(sentence, lang_code, method)
                for lang_code in lang_codes
            }
            if data:
                entry |= data
            results.append(entry)
            cache[sentence] = entry
            pending_sentences.append(sentence)
            pending_entries.append(entry)
            new_count += 1

    if pending_sentences:
        translated_by_lang = _translate_many_googletrans(pending_sentences, lang_codes)
        for i, entry in enumerate(tqdm(pending_entries)):
            for lang_code in lang_codes:
                translated = translated_by_lang[lang_code][i]
                entry[lang_code] = translated.text
                entry[f"{lang_code}_en"] = unidecode(translated.pronunciation)

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
