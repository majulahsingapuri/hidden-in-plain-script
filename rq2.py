import argparse
from json import load, dump
from tqdm import tqdm
from nnsight import LanguageModel
from datetime import datetime
from huggingface_hub import login
from pathlib import Path

from config import Config
from utils import (
    build_variants,
    count_batches,
    count_work_items,
    generate_trace,
    iter_batches,
    iter_work_items,
    ResourceMonitor,
)


def run_experiment(
    data_path: Path,
    model_name: str,
    layers_path: str,
    norm_path: str,
    langs: list[str],
    limit: int = 0,
    batch_size: int = 16,
):
    config = Config()
    print("Config Loaded")

    login(token=config.hf_token)
    print("Logged into HF")

    model = LanguageModel(model_name, device_map="auto", dispatch=True)
    print(f"Model loaded: {model_name}")

    total_results = []

    with open(data_path, "r") as f:
        prompts = load(f)

    if limit:
        prompts = prompts[:limit]

    variants = build_variants(langs)

    total_items = count_work_items(prompts, variants)
    total_batches = count_batches(total_items, batch_size)
    work_iter = iter_work_items(prompts, variants)
    monitor = ResourceMonitor()
    pbar = tqdm(
        iter_batches(work_iter, batch_size),
        total=total_batches,
        position=0,
        desc="Generating Data",
    )
    for batch_items in pbar:
        postfix = monitor.tqdm_postfix()
        if postfix:
            pbar.set_postfix(postfix, refresh=False)
        batch_inputs = [item["prompt_text"] for item in batch_items]

        responses = generate_trace(
            model,
            batch_inputs,
            layers_path=layers_path,
            norm_path=norm_path,
        )

        for item, response  in zip(
            batch_items, responses, strict=True
        ):
            result = {
                "variant": item["variant"],
                "prompt": item["prompt_text"],
                "prompt_id": item["prompt_id"],
                "words": response["words"],
                "tokens": response["tokens"].tolist()
            }
            total_results.append(result)

    monitor.close()

    save_path = Path( 
        f"./results/rq2/{model_name.replace("/", "-")}"
        + f"-{datetime.now().isoformat()}.json" 
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

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
        default=0,
        help="Limit the number of data points used for run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--layers-path",
        type=str,
        default="model.language_model.layers",
        help="Dot path from the LanguageModel object to the layer list/ModuleList.",
    )
    parser.add_argument(
        "--norm-path",
        type=str,
        default="model.language_model.norm",
        help="Dot path from the LanguageModel object to the final norm module.",
    )
    parser.add_argument(
        "--langs",
        "-l",
        nargs="+",
        default=["gu", "hi", "ta", "te"],
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
        args.data,
        args.model,
        args.layers_path,
        args.norm_path,
        args.langs,
        args.limit,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
