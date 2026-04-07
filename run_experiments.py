import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-search sweep for train_classifier.py."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing sae_features.json and sae_activations/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sweep results.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of sweep trials.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_f1",
        help="Metric to optimize (val_f1, val_roc_auc, val_loss).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to pass through (e.g., mps, cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for sweep and training splits.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of activation rows used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs for each trial.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for each trial.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for train_classifier.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned configs, do not run training.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path | None) -> Path:
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("results") / "rq3" / "sweeps" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def sample_config(rng: random.Random) -> dict[str, Any]:
    return {
        "lr": log_uniform(rng, 1e-4, 5e-3),
        "weight_decay": log_uniform(rng, 1e-6, 1e-2),
        "top_k": rng.choice([0, 512, 1024, 2048, 4096, 8192]),
        "normalize": rng.choice(["zscore", "l2", "none"]),
        "max_grad_norm": rng.choice([0.0, 1.0, 5.0]),
        "batch_size": rng.choice([64, 128, 256]),
    }


def metric_from_metrics(metrics: dict[str, Any], metric_name: str) -> float | None:
    metric_name = metric_name.lower()
    if metric_name == "val_f1":
        return metrics.get("val", {}).get("f1")
    if metric_name == "val_roc_auc":
        return metrics.get("val", {}).get("roc_auc")
    if metric_name == "val_loss":
        return metrics.get("val", {}).get("loss")
    return None


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def build_cmd(
    args: argparse.Namespace,
    output_root: Path,
    model_name: str,
    config: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        "train_classifier.py",
        "--data-dir",
        str(args.data_dir),
        "--output-root",
        str(output_root),
        "--model-name",
        model_name,
        "--lr",
        str(config["lr"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--top-k",
        str(config["top_k"]),
        "--normalize",
        str(config["normalize"]),
        "--max-grad-norm",
        str(config["max_grad_norm"]),
        "--batch-size",
        str(args.batch_size or config["batch_size"]),
        "--seed",
        str(args.seed),
        "--log-level",
        args.log_level,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.max_items:
        cmd.extend(["--max-items", str(args.max_items)])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    return cmd


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    rng = random.Random(args.seed)

    results: list[dict[str, Any]] = []
    best_entry: dict[str, Any] | None = None
    best_value: float | None = None

    for trial in range(1, args.trials + 1):
        config = sample_config(rng)
        model_name = f"trial-{trial:03d}"
        cmd = build_cmd(args, output_dir, model_name, config)
        trial_dir = output_dir / model_name
        entry: dict[str, Any] = {
            "trial": trial,
            "model_name": model_name,
            "config": config,
            "cmd": cmd,
            "status": "planned",
            "output_dir": str(trial_dir),
        }

        if args.dry_run:
            results.append(entry)
            continue

        try:
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                entry["status"] = "failed"
                entry["error"] = f"train_classifier exited with {proc.returncode}"
            else:
                metrics_path = trial_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    entry["metrics"] = metrics
                    metric_value = metric_from_metrics(metrics, args.metric)
                    entry["metric_value"] = metric_value
                    entry["status"] = "ok"
                    if metric_value is not None:
                        if best_value is None:
                            best_value = metric_value
                            best_entry = entry
                        else:
                            if args.metric == "val_loss":
                                if metric_value < best_value:
                                    best_value = metric_value
                                    best_entry = entry
                            else:
                                if metric_value > best_value:
                                    best_value = metric_value
                                    best_entry = entry
                else:
                    entry["status"] = "failed"
                    entry["error"] = "metrics.json not found"
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)

        results.append(entry)

        save_json(output_dir / "results.json", results)
        if best_entry is not None:
            save_json(output_dir / "best.json", best_entry)

    if not args.dry_run:
        save_json(output_dir / "results.json", results)
        if best_entry is not None:
            save_json(output_dir / "best.json", best_entry)


if __name__ == "__main__":
    main()
