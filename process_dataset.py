from datasets import load_dataset
from transliterate import main as tranliterate_main
from pathlib import Path
from json import dump
import argparse
from random import shuffle


def download_dataset(path: Path, limit: int = None):
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")

    harmful_prompts = [
        {
            "prompt_id": f"harmful_{row['Index']}",
            "en": row["Goal"],
            "behavior": row["Behavior"],
            "split": "harmful",
        }
        for row in jbb["harmful"]
    ]

    benign_prompts = [
        {
            "prompt_id": f"benign_{row['Index']}",
            "en": row["Goal"],
            "behavior": row["Behavior"],
            "split": "benign",
        }
        for row in jbb["benign"]
    ]

    prompts = harmful_prompts + benign_prompts

    if limit:
        shuffle(prompts)
        prompts = prompts[:limit]

    with open(path, "w") as f:
        dump(prompts, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Transliterate sentences into one or more target languages."
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
        "--limit",
        type=int,
        help="limit the number of datapoints processed",
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

    args = parser.parse_args()
    args.sentences_file = args.output_path

    download_dataset(args.output_path, args.limit)

    tranliterate_main(args)


if __name__ == "__main__":
    main()
