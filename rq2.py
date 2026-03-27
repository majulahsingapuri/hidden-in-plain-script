import argparse
from json import load, dump
from tqdm import tqdm
from nnsight import LanguageModel
from datetime import datetime
from huggingface_hub import login
from pathlib import Path

from config import Config
from utils import generate_trace


def run_experiment(
    data_path: Path,
    langs: list[str],
    limit: int = 5,
    batch_size: int = 16,
):
    config = Config()
    print("Config Loaded")

    login(token=config.hf_token)
    print("Logged into HF")

    model = LanguageModel(config.model, device_map="auto", dispatch=True)
    print(f"Model loaded: {config.model}")

    total_results = []

    with open(data_path, "r") as f:
        prompts = load(f)

    if limit:
        prompts = prompts[:limit]

    variants = ["en"] + [
        version for lang in langs for version in [lang, f"{lang}_en", f"en_{lang}"]
    ]

    work_items = []
    for prompt in prompts:
        for variant in variants:
            work_items.append(
                {
                    "variant": variant,
                    "prompt_text": prompt[variant],
                    "prompt_en": prompt["en"],
                }
            )

    batches = []
    for i in tqdm(range(0, len(work_items), batch_size), position=0, desc="Batching Inputs"):
        batch_items = work_items[i : i + batch_size]
        batches.append((batch_items, [item["prompt_text"] for item in batch_items]))

    for batch_items, batch_inputs in tqdm(batches, position=0, desc="Generating Data"):

        responses = generate_trace(model, batch_inputs)

        for item, response  in zip(
            batch_items, responses, strict=True
        ):
            result = {
                "variant": item["variant"],
                "prompt": item["prompt_text"],
                "words": response["words"],
                "tokens": response["tokens"].tolist()
            }
            total_results.append(result)

    save_path = Path( 
        f"./results/rq2/{config.model.replace("/", "-")}"
        + f"-{config.judge_model_name}-{config.temperature}"
        + f"-{config.top_p}-{datetime.now().isoformat()}.json" 
    ).mkdir(parents=True, exist_ok=True)

    with open(
        save_path,
        "w",
    ) as f:
        dump(total_results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1.")
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default=Path.cwd() / "assets" / "transliterations.json",
        help="Path to a JSON file containing dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of data points used for run",
    )
    parser.add_argument(
        "--langs",
        "-l",
        nargs="+",
        default=["gu"],
        help="Target language codes (e.g., gu te).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for preprocessing and generation.",
    )
    args = parser.parse_args()
    run_experiment(
        args.data, args.langs, args.limit, args.batch_size
    )


if __name__ == "__main__":
    main()
