import json
import os
from google.transliteration import transliterate_word


OUTPUT_PATH = "assets/transliterations.json"


def transliterate_sentence(sentence: str, lang_code: str) -> str:
    """Transliterate each word in a sentence to the target language."""
    words = sentence.split()
    transliterated = []
    for word in words:
        results = transliterate_word(word, lang_code=lang_code)
        transliterated.append(results[0] if results else word)
    return " ".join(transliterated)


def load_existing(path: str) -> list[dict]:
    """Load transliterations from an existing JSON file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_to_json(data: list[dict], path: str):
    """Write the full data list to the JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_sentences(sentences: list[str], path: str) -> list[dict]:
    """
    For each sentence:
      - If already in the JSON cache, reuse it (no API call).
      - If missing, transliterate it and append to the cache.
    Saves the updated JSON at the end if anything new was added.
    """
    existing = load_existing(path)

    # Build a lookup dict keyed by the English sentence
    cache = {entry["english"]: entry for entry in existing}

    results = []
    new_count = 0

    for sentence in sentences:
        if sentence in cache:
            print(f"[CACHE]  {sentence}")
            results.append(cache[sentence])
        else:
            print(f"[NEW]    {sentence}")
            telugu = transliterate_sentence(sentence, lang_code="te")
            gujarati = transliterate_sentence(sentence, lang_code="gu")
            entry = {
                "english": sentence,
                "telugu": telugu,
                "gujarati": gujarati,
            }
            results.append(entry)
            cache[sentence] = entry
            new_count += 1

    # Save only if there were new transliterations
    if new_count > 0:
        all_entries = list(cache.values())
        save_to_json(all_entries, path)
        print(f"\nAdded {new_count} new entries. Total: {len(all_entries)} in {path}")
    else:
        print(f"\nAll {len(results)} sentences found in cache. No API calls needed.")

    # Print all results
    print("\n--- Results ---\n")
    for entry in results:
        print(f"  English : {entry['english']}")
        print(f"  Telugu  : {entry['telugu']}")
        print(f"  Gujarati: {entry['gujarati']}")
        print()

    return results


def process_prompts(prompts):
    for x in prompts:
        tegu = process_sentences([x['goal']], OUTPUT_PATH)[0]
        x['gujarati'] = tegu['gujarati']
        x['telugu'] = tegu['telugu']
    return prompts


print(process_prompts([{"goal":"Yolo what is up developer?"}]))