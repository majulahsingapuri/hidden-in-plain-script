import argparse
from json import load, dump
from tqdm import tqdm
from nnsight import LanguageModel
from datetime import datetime
from pydantic_ai import Agent
from huggingface_hub import login
from pathlib import Path

from judge import judge_response, JUDGE_SYSTEM_PROMPT, EvaluationResponse
from config import Config
from utils import generate_response


def run_experiment(data_path: Path, langs: list[str], limit: int = 5):
    config = Config()
    print("Config Loaded")

    login(token=config.hf_token)
    print("Logged into HF")

    model = LanguageModel(config.model, device_map="auto", dispatch=True)
    print(f"Model loaded: {config.model}")

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

    for prompt in tqdm(prompts):
        for variant in variants:
            response = generate_response(
                model,
                prompt[variant],
            )
            verdict = judge_response(judge, prompt["en"], response)

            result = {
                "variant": variant,
                "prompt": prompt[variant],
                "response": response,
                "judge_refused": verdict.refused,
                "judge_harmful": verdict.harmful,
                "judge_reason": verdict.explanation,
            }
            total_results.append(result)

    with open(
        f"results/{config.model}-{config.judge_model_name}-{config.temperature}"
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
    args = parser.parse_args()
    run_experiment(args.data, args.langs, args.limit)


if __name__ == "__main__":
    main()
