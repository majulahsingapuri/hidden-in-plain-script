"""Inference utilities for SAE-based harmfulness classifier.

Sample CLI usage:
  python run_classifier.py --model results/rq3/gemma-3-4b-it \
    --activation assets/gemma-3-4b-it/sae_activations/benign_0-en-17.pt

  python run_classifier.py --model results/rq3/gemma-3-4b-it \
    --data-dir assets/gemma-3-4b-it --example harmful_0:en

Sample notebook usage:
  from run_classifier import ClassifierRunner
  runner = ClassifierRunner(
      Path("results/rq3/gemma-3-4b-it/classifier.pt"),
      config_path=Path("results/rq3/gemma-3-4b-it/config.json"),
  )
  result = runner.predict_tensor(torch.load(".../benign_0-en-17.pt"))
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def resolve_activation_path(data_dir: Path, activation_path: str) -> Path:
    """Resolve an activation path recorded in `sae_features.json`.

    Example:
        >>> # resolve_activation_path(Path("assets/gemma-3-4b-it"), "benign_0-en-17.pt")
    """
    path = Path(activation_path)
    if path.exists():
        return path
    return data_dir / "sae_activations" / path.name


def apply_top_k(x: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """Keep only the top-k features in a single activation vector."""
    if k <= 0 or k >= x.numel():
        return x
    if mode == "abs":
        scores = x.abs()
    elif mode == "pos":
        scores = x
    else:
        raise ValueError(f"Unknown top-k mode: {mode}")
    _, idx = torch.topk(scores, k=k, largest=True, sorted=False)
    out = torch.zeros_like(x)
    out[idx] = x[idx]
    return out


def apply_top_k_batch(x: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """Keep only the top-k features for each row in a 2D batch tensor."""
    if k <= 0 or k >= x.size(1):
        return x
    if mode == "abs":
        scores = x.abs()
    elif mode == "pos":
        scores = x
    else:
        raise ValueError(f"Unknown top-k mode: {mode}")
    _, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=False)
    out = torch.zeros_like(x)
    out.scatter_(1, idx, x.gather(1, idx))
    return out


@dataclass
class Normalizer:
    """Base class for feature normalization."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one activation vector."""

        raise NotImplementedError


@dataclass
class ZScoreNormalizer(Normalizer):
    """Per-feature z-score normalization."""
    mean: torch.Tensor
    std: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization using stored statistics."""

        return (x - self.mean) / self.std


@dataclass
class L2Normalizer(Normalizer):
    """L2 normalization per vector."""
    eps: float = 1e-8

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Scale a vector to unit L2 norm."""

        return x / (x.norm(p=2) + self.eps)


class ClassifierRunner:
    """Load a trained classifier and run inference on activations.

    Example:
        >>> from pathlib import Path
        >>> runner = ClassifierRunner(Path("results/rq3/gemma-3-4b-it/classifier.pt"))
        >>> # runner.predict_file(Path("assets/gemma-3-4b-it/sae_activations/benign_0-en-17.pt"))
    """

    def __init__(self, model_path: Path, config_path: Path | None = None):
        """Create a runner from `classifier.pt` and optional `config.json`."""
        self.model_path = model_path
        payload = torch.load(model_path, map_location="cpu")
        config = None
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        self.input_dim = int(payload["input_dim"])
        self.threshold = float(payload.get("threshold", 0.5))
        self.temperature = float(payload.get("temperature", 1.0))
        self.top_k = int(payload.get("top_k", 0) or 0)
        self.top_k_mode = payload.get("top_k_mode", "abs")

        if config is not None:
            if config.get("temperature") is not None:
                self.temperature = float(config["temperature"])
            if config.get("top_k") is not None:
                self.top_k = int(config["top_k"])
            if config.get("top_k_mode") is not None:
                self.top_k_mode = str(config["top_k_mode"])

        self.normalizer = self._build_normalizer(
            payload.get("normalizer"),
            config.get("normalizer") if config is not None else None,
        )
        self.model = nn.Linear(self.input_dim, 1)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def _build_normalizer(
        self,
        normalizer_state: dict | None,
        config_normalizer: dict | None,
    ) -> Normalizer | None:
        """Build a normalizer from checkpoint and config metadata."""
        if config_normalizer and config_normalizer.get("type") == "zscore":
            path = config_normalizer.get("path")
            if path:
                payload = torch.load(path, map_location="cpu")
                mean = payload.get("mean")
                std = payload.get("std")
                if mean is not None and std is not None:
                    mean_t = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean)
                    std_t = std if isinstance(std, torch.Tensor) else torch.tensor(std)
                    return ZScoreNormalizer(mean_t.float(), std_t.float())
        if not normalizer_state:
            return None
        ntype = normalizer_state.get("type")
        if ntype == "zscore":
            mean = normalizer_state.get("mean")
            std = normalizer_state.get("std")
            if mean is None or std is None:
                return None
            mean_t = mean if isinstance(mean, torch.Tensor) else torch.tensor(mean)
            std_t = std if isinstance(std, torch.Tensor) else torch.tensor(std)
            return ZScoreNormalizer(mean_t.float(), std_t.float())
        if ntype == "l2":
            return L2Normalizer(float(normalizer_state.get("eps", 1e-8)))
        return None

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and top-k filter a 1D or 2D tensor into `[B, D]`."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Input shape {tuple(x.shape)} does not match (*, {self.input_dim})"
            )
        x = x.float()
        if self.normalizer is not None:
            if isinstance(self.normalizer, ZScoreNormalizer):
                x = (x - self.normalizer.mean) / self.normalizer.std
            elif isinstance(self.normalizer, L2Normalizer):
                x = x / (x.norm(p=2, dim=1, keepdim=True) + self.normalizer.eps)
            else:
                x = torch.stack([self.normalizer(row) for row in x], dim=0)
        if self.top_k and self.top_k > 0:
            x = apply_top_k_batch(x, self.top_k, self.top_k_mode)
        return x

    def predict_batch_raw(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw batched outputs as tensors.

        The returned dictionary contains `logits`, `probs`, and `preds`.
        """
        x = self._prep(x)
        with torch.no_grad():
            logits = self.model(x).squeeze(1)
        probs = torch.sigmoid(logits / self.temperature)
        preds = (probs >= self.threshold).int()
        return {"logits": logits, "probs": probs, "preds": preds}

    def predict_tensor(self, x: torch.Tensor) -> dict[str, Any] | list[dict[str, Any]]:
        """Return human-friendly scalar predictions for one tensor or a batch."""
        outputs = self.predict_batch_raw(x)
        logits = outputs["logits"].tolist()
        probs = outputs["probs"].tolist()
        preds = outputs["preds"].tolist()
        results = []
        for logit, prob, pred in zip(logits, probs, preds, strict=False):
            results.append(
                {
                    "logit": float(logit),
                    "prob": float(prob),
                    "pred": int(pred),
                    "threshold": self.threshold,
                    "temperature": self.temperature,
                }
            )
        return results[0] if len(results) == 1 else results

    def predict_file(self, path: Path) -> dict[str, Any]:
        """Load one activation `.pt` file and return its prediction dictionary."""
        vec = torch.load(path, map_location="cpu")
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec)
        result = self.predict_tensor(vec)
        if isinstance(result, list):
            if len(result) == 1:
                return result[0]
            raise ValueError("predict_file expects a single activation vector.")
        return result


def load_examples_index(data_dir: Path) -> dict[tuple[str, str], str]:
    """Build an index from `(prompt_id, variant)` to activation path."""
    features_path = data_dir / "sae_features.json"
    with open(features_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("sae_features.json must be a list of dicts.")
    index: dict[tuple[str, str], str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("prompt_id"))
        variant = str(item.get("variant"))
        activation_path = str(item.get("activation_path"))
        index[(pid, variant)] = activation_path
    return index


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch classifier inference."""
    parser = argparse.ArgumentParser(description="Run inference on a trained classifier.")
    parser.add_argument(
        "--model",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to classifier .pt file(s) or directories containing classifier.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.json to override model parameters.",
    )
    parser.add_argument(
        "--activation",
        type=Path,
        nargs="+",
        default=[],
        help="Path(s) to activation .pt files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data dir containing sae_features.json and sae_activations/.",
    )
    parser.add_argument(
        "--example",
        type=str,
        nargs="+",
        default=[],
        help="Example keys as prompt_id:variant (requires --data-dir).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file for JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the classifier CLI and emit JSON lines.

    Example:
        ```bash
        python run_classifier.py --model results/rq3/gemma-3-4b-it --example harmful_0:en --data-dir assets/gemma-3-4b-it
        ```
    """
    args = parse_args()
    outputs: list[str] = []

    example_inputs: list[tuple[str | None, str | None, Path]] = []
    if args.example:
        if args.data_dir is None:
            raise ValueError("--data-dir is required when using --example")
        index = load_examples_index(args.data_dir)
        for ex in args.example:
            if ":" not in ex:
                raise ValueError(f"Invalid example format: {ex}")
            prompt_id, variant = ex.split(":", 1)
            activation_path = index.get((prompt_id, variant))
            if activation_path is None:
                raise ValueError(f"Example not found: {ex}")
            path = resolve_activation_path(args.data_dir, activation_path)
            example_inputs.append((prompt_id, variant, path))

    for path in args.activation:
        example_inputs.append((None, None, path))

    for model_path in args.model:
        config_path = args.config
        model_file = model_path
        if model_path.is_dir():
            model_file = model_path / "classifier.pt"
            if config_path is None:
                candidate = model_path / "config.json"
                if candidate.exists():
                    config_path = candidate
        runner = ClassifierRunner(model_file, config_path=config_path)
        for prompt_id, variant, path in example_inputs:
            result = runner.predict_file(path)
            record = {
                "model_path": str(model_file),
                "input_path": str(path),
                "prompt_id": prompt_id,
                "variant": variant,
                **result,
            }
            outputs.append(json.dumps(record))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(outputs))
            if outputs:
                f.write("\n")
    else:
        for line in outputs:
            print(line)


if __name__ == "__main__":
    main()
