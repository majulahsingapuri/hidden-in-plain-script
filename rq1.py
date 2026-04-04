import argparse
import asyncio
from json import load, dump
from tqdm import tqdm
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
from datetime import datetime
from pydantic_ai import Agent
from huggingface_hub import login
from pathlib import Path
import torch

from judge import judge_response_async, JUDGE_SYSTEM_PROMPT, EvaluationResponse
from config import Config
from utils import (
    build_variants,
    count_batches,
    count_work_items,
    iter_batches,
    iter_work_items,
    ResourceMonitor,
)


async def judge_batch(
    judge: Agent, batch_items: list[dict], responses: list[str], concurrency: int
):
    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(item: dict, response: str):
        async with semaphore:
            return await judge_response_async(judge, item["prompt_en"], response)

    tasks = [
        asyncio.create_task(_judge_one(item, response))
        for item, response in zip(batch_items, responses, strict=True)
    ]
    return await asyncio.gather(*tasks)


def run_experiment(
    data_path: Path,
    model_name: str,
    langs: list[str],
    limit: int = 0,
    batch_size: int = 16,
    judge_concurrency: int = 4,
    output_path: Path | None = None,
    resume: bool = False,
):
    config = Config()
    print("Config Loaded")

    login(token=config.hf_token)
    print("Logged into HF")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batched generation but is not available.")
    device = torch.device("cuda")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    processor = Gemma3Processor.from_pretrained(model_name)
    print(f"Model loaded: {model_name}")

    judge = Agent(
        f"{config.judge_provider}:{config.judge_model_name}",
        system_prompt=JUDGE_SYSTEM_PROMPT,
        output_type=EvaluationResponse,
    )
    print("Judge Initialised")

    save_path = (
        output_path
        if output_path is not None
        else Path(
            f"./results/rq1/{model_name.replace("/", "-")}"
            + f"-{config.judge_model_name}"
            + f"-{datetime.now().isoformat()}.json"
        )
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total_results = []
    completed_keys: set[tuple[str, str]] = set()
    if resume and save_path.exists():
        with open(save_path, "r") as f:
            try:
                total_results = load(f)
            except Exception:
                total_results = []
        if isinstance(total_results, list):
            completed_keys = {
                (str(item.get("prompt_id")), str(item.get("variant")))
                for item in total_results
                if isinstance(item, dict)
                and item.get("prompt_id") is not None
                and item.get("variant") is not None
            }
        else:
            total_results = []
            completed_keys = set()
        print(f"Resuming: loaded {len(completed_keys)} completed items from {save_path}")
    elif resume:
        print(f"Resume requested but no existing file at {save_path}. Starting fresh.")

    with open(data_path, "r") as f:
        prompts = load(f)

    if limit:
        prompts = prompts[:limit]

    variants = build_variants(langs)

    total_items = count_work_items(prompts, variants, completed_keys)
    total_batches = count_batches(total_items, batch_size)
    work_iter = iter_work_items(prompts, variants, completed_keys)
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
        batch_inputs = processor(
            text=[item["prompt_text"] for item in batch_items],
            padding=True,
            return_tensors="pt",
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=300,
            do_sample=False,
        )
        responses = processor.batch_decode(outputs, skip_special_tokens=True)

        verdicts = asyncio.run(
            judge_batch(judge, batch_items, responses, judge_concurrency)
        )
        for item, response, verdict in zip(
            batch_items, responses, verdicts, strict=True
        ):
            result = {
                "variant": item["variant"],
                "prompt": item["prompt_text"],
                "prompt_id": item["prompt_id"],
                "response": response,
                "refused": verdict.refused,
                "harmful": verdict.harmful,
                "gibberish": verdict.gibberish,
            }
            total_results.append(result)
            completed_keys.add((str(item["prompt_id"]), str(item["variant"])))

        with open(save_path, "w") as f:
            dump(total_results, f, indent=2, ensure_ascii=False)

    monitor.close()

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
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of data points used for run",
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
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=4,
        help="Max number of concurrent judge LLM calls.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to write results JSON. If omitted, a timestamped path is used.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output file by skipping completed prompt_id+variant pairs.",
    )
    args = parser.parse_args()
    run_experiment(
        args.data,
        args.model,
        args.langs,
        args.limit,
        args.batch_size,
        args.judge_concurrency,
        args.output,
        args.resume,
    )


if __name__ == "__main__":
    main()
