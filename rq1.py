import argparse
from json import load, dump
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
from datetime import datetime
from pydantic_ai import Agent
from huggingface_hub import login
from pathlib import Path
import torch

from judge import judge_response, JUDGE_SYSTEM_PROMPT, EvaluationResponse
from config import Config


def run_experiment(
    data_path: Path, langs: list[str], limit: int = 5, batch_size: int = 16
):
    config = Config()
    print("Config Loaded")

    login(token=config.hf_token)
    print("Logged into HF")

    model = Gemma3ForConditionalGeneration.from_pretrained(config.model)
    processor = Gemma3Processor.from_pretrained(config.model)
    print(f"Model loaded: {config.model}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batched generation but is not available.")
    device = torch.device("cuda")
    model.to(device)

    judge = Agent(
        f"{config.judge_provider}:{config.judge_model_name}",
        system_prompt=JUDGE_SYSTEM_PROMPT,
        output_type=EvaluationResponse,
    )
    print("Judge Initialised")

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
    for i in tqdm(range(0, len(work_items), batch_size), position=0):
        batch_items = work_items[i : i + batch_size]
        batch_inputs = processor(
            text=[item["prompt_text"] for item in batch_items],
            padding=True,
            return_tensors="pt",
        )
        batches.append((batch_items, batch_inputs))

    for batch_items, batch_inputs in tqdm(batches, position=0):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=100,
            do_sample=False,
        )
        responses = processor.batch_decode(outputs, skip_special_tokens=False)

        for item, response in zip(batch_items, responses, strict=True):
            verdict = judge_response(judge, item["prompt_en"], response)

            result = {
                "variant": item["variant"],
                "prompt": item["prompt_text"],
                "response": response,
                "refused": verdict.refused,
                "harmful": verdict.harmful,
                "gibberish": verdict.gibberish,
            }
            total_results.append(result)

    with open(
        f"./results/{config.model.replace("/", "-")}-{config.judge_model_name}-{config.temperature}"
        + f"-{config.top_p}-{datetime.now().isoformat()}.json",
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
    run_experiment(args.data, args.langs, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
