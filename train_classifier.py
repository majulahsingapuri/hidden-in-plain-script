"""Train a linear probe on SAE activations.

Example:
    ```bash
    python train_classifier.py --data-dir assets/gemma-3-4b-it --model-name gemma-3-4b-it
    ```
"""

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Example:
    """Metadata for one activation example used during classifier training."""

    prompt_id: str
    variant: str
    target_layer: int
    activation_path: str
    label: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for probe training."""

    parser = argparse.ArgumentParser(
        description="Train a linear probe on SAE activations to detect harmful prompts."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing sae_features.json and sae_activations/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "rq3",
        help="Root output directory.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Output folder name. Defaults to basename of --data-dir.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help="Optional target layer filter.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of activation rows used.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument(
        "--normalize",
        type=str,
        default="zscore",
        choices=["none", "zscore", "l2"],
        help="Feature normalization mode.",
    )
    parser.add_argument(
        "--l2-eps",
        type=float,
        default=1e-8,
        help="Epsilon for L2 normalization.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Clip gradient norm (0 to disable).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4096,
        help="Keep only top-k features per example (0 to disable).",
    )
    parser.add_argument(
        "--top-k-mode",
        type=str,
        default="abs",
        choices=["abs", "pos"],
        help="Top-k selection mode: abs (by absolute value) or pos (by value).",
    )
    parser.set_defaults(temperature_scale=True)
    parser.add_argument(
        "--no-temperature-scale",
        action="store_false",
        dest="temperature_scale",
        help="Disable temperature scaling.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop if val F1 doesn't improve for this many epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum F1 improvement to reset early stopping.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold for classification. If unset, tune on val.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def resolve_activation_path(data_dir: Path, activation_path: str) -> Path:
    """Resolve an activation path relative to the data directory if needed."""

    path = Path(activation_path)
    if path.exists():
        return path
    # Fall back to data_dir/sae_activations/<filename>
    return data_dir / "sae_activations" / path.name


def label_from_prompt_id(prompt_id: str) -> int:
    """Infer the binary label from the prompt ID prefix."""

    if prompt_id.startswith("harmful_"):
        return 1
    if prompt_id.startswith("benign_"):
        return 0
    raise ValueError(f"Unknown prompt_id prefix for label: {prompt_id}")


def load_examples(
    data_dir: Path, target_layer: int | None = None, max_items: int | None = None
) -> list[Example]:
    """Load activation metadata rows from `sae_features.json`.

    Example:
        >>> # examples = load_examples(Path("assets/gemma-3-4b-it"))
    """

    features_path = data_dir / "sae_features.json"
    logging.info("Loading features from %s", features_path)
    with open(features_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("sae_features.json must be a list of dicts.")

    examples: list[Example] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        layer = item.get("target_layer")
        if target_layer is not None and layer != target_layer:
            continue
        prompt_id = str(item.get("prompt_id"))
        variant = str(item.get("variant"))
        activation_path = str(item.get("activation_path"))
        label = label_from_prompt_id(prompt_id)
        examples.append(
            Example(
                prompt_id=prompt_id,
                variant=variant,
                target_layer=int(layer) if layer is not None else -1,
                activation_path=activation_path,
                label=label,
            )
        )
        if max_items is not None and len(examples) >= max_items:
            break
    if not examples:
        raise ValueError("No examples found after filtering.")
    logging.info("Loaded %d examples", len(examples))
    return examples


def stratified_split(
    examples: list[Example], seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Create reproducible train/validation/test splits per class label."""

    label_to_indices: dict[int, list[int]] = {0: [], 1: []}
    for idx, ex in enumerate(examples):
        label_to_indices[ex.label].append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label, indices in label_to_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        logging.debug("Label %s count: %d", label, n)
        n_train = int(n * 0.7)
        n_val = int(n * 0.2)
        n_test = n - n_train - n_val
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    logging.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return train_idx, val_idx, test_idx


class ActivationDataset(Dataset):
    """PyTorch dataset that loads activation vectors on demand."""

    def __init__(
        self,
        data_dir: Path,
        examples: list[Example],
        indices: list[int],
        normalizer: "Normalizer | None" = None,
        top_k: int | None = None,
        top_k_mode: str = "abs",
    ):
        """Store activation metadata and preprocessing settings."""

        self.data_dir = data_dir
        self.examples = examples
        self.indices = indices
        self.normalizer = normalizer
        self.top_k = top_k
        self.top_k_mode = top_k_mode

    def __len__(self) -> int:
        """Return the number of examples in this split."""

        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load, normalize, sparsify, and label one activation vector."""

        ex = self.examples[self.indices[idx]]
        path = resolve_activation_path(self.data_dir, ex.activation_path)
        if not path.exists():
            raise FileNotFoundError(f"Activation file not found: {path}")
        vec = torch.load(path, map_location="cpu")
        if isinstance(vec, torch.Tensor):
            tensor = vec
        else:
            tensor = torch.tensor(vec)
        tensor = tensor.float().flatten()
        if self.normalizer is not None:
            tensor = self.normalizer(tensor)
        if self.top_k is not None and self.top_k > 0:
            tensor = apply_top_k(tensor, self.top_k, self.top_k_mode)
        return tensor, ex.label


def collate_batch(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate activation vectors and labels into tensors for a `DataLoader`."""

    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.float32)


class Normalizer:
    """Base interface for activation normalizers."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize one activation vector."""

        raise NotImplementedError


class ZScoreNormalizer(Normalizer):
    """Per-feature z-score normalization using saved train-set statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Store the mean and standard deviation tensors."""

        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization to one vector."""

        return (x - self.mean) / self.std


class L2Normalizer(Normalizer):
    """Normalize each vector to unit L2 norm."""

    def __init__(self, eps: float = 1e-8):
        """Store the epsilon used to avoid division by zero."""

        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalization to one vector."""

        return x / (x.norm(p=2) + self.eps)


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


def compute_zscore_stats(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and standard deviation for a dataset."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    sum_ = None
    sumsq = None
    count = 0
    for xs, _ in tqdm(loader, desc="Compute zscore stats", leave=False):
        xs = xs.double()
        if sum_ is None:
            sum_ = xs.sum(dim=0)
            sumsq = (xs * xs).sum(dim=0)
        else:
            sum_ += xs.sum(dim=0)
            sumsq += (xs * xs).sum(dim=0)
        count += xs.size(0)
    if sum_ is None or sumsq is None or count == 0:
        raise RuntimeError("Failed to compute normalization stats.")
    mean = sum_ / count
    var = sumsq / count - mean * mean
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)
    return mean.float(), std.float()


def compute_metrics(
    y_true: list[int], y_prob: list[float], threshold: float
) -> dict[str, float]:
    """Compute classification metrics from probabilities and a threshold."""

    preds = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 0)

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / max(1e-8, precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(len(y_true)),
    }

    try:
        from sklearn.metrics import roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return metrics


def collect_logits(
    model: nn.Module, loader: DataLoader, device: torch.device, desc: str | None = None
) -> tuple[list[int], list[float]]:
    """Run the model over a loader and collect labels plus raw logits."""

    model.eval()
    y_true: list[int] = []
    y_logits: list[float] = []
    with torch.no_grad():
        iterable = loader
        if desc:
            iterable = tqdm(loader, desc=desc, leave=False)
        for xs, ys in iterable:
            xs = xs.to(device)
            ys = ys.to(device)
            logits = model(xs).squeeze(1)
            y_logits.extend([float(x) for x in logits.detach().cpu().tolist()])
            y_true.extend([int(y) for y in ys.detach().cpu().tolist()])
    return y_true, y_logits


def logits_to_probs(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Convert logits to sigmoid probabilities with optional temperature scaling."""

    logits_t = torch.tensor(logits, dtype=torch.float32) / float(temperature)
    probs = torch.sigmoid(logits_t).tolist()
    return [float(p) for p in probs]


def compute_bce_loss(
    y_true: list[int], logits: list[float], temperature: float = 1.0
) -> float:
    """Compute binary cross-entropy loss from Python lists of labels and logits."""

    if not y_true:
        return 0.0
    logits_t = torch.tensor(logits, dtype=torch.float32) / float(temperature)
    labels_t = torch.tensor(y_true, dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss()(logits_t, labels_t)
    return float(loss.item())


def fit_temperature(y_true: list[int], logits: list[float]) -> float:
    """Fit a scalar temperature on validation logits using LBFGS."""

    if len(set(y_true)) < 2:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(y_true, dtype=torch.float32)
    log_t = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        """LBFGS closure that evaluates calibration loss for the current temperature."""

        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = nn.BCEWithLogitsLoss()(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).item())
    return max(temperature, 1e-3)


def tune_threshold(y_true: list[int], y_prob: list[float]) -> float:
    """Choose the threshold in `[0, 1]` that maximizes F1 on validation data."""

    best_thresh = 0.5
    best_f1 = -1.0
    for i in range(0, 101):
        t = i / 100.0
        metrics = compute_metrics(y_true, y_prob, t)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = t
    return best_thresh


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    desc: str | None = None,
    max_grad_norm: float = 0.0,
) -> float:
    """Run one training epoch and return the average batch loss."""

    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_count = 0
    iterable = loader
    if desc:
        iterable = tqdm(loader, desc=desc, leave=False)
    for xs, ys in iterable:
        xs = xs.to(device)
        ys = ys.to(device)
        optimizer.zero_grad()
        logits = model(xs).squeeze(1)
        loss = loss_fn(logits, ys)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += float(loss.item()) * xs.size(0)
        total_count += xs.size(0)
    return total_loss / max(1, total_count)


def make_output_dir(output_root: Path, model_name: str) -> Path:
    """Create the training output directory for a model run."""

    out_dir = output_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_json(path: Path, payload: Any):
    """Write JSON with UTF-8 encoding and stable indentation."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    """Run classifier training from the command line.

    Example:
        ```bash
        python train_classifier.py --data-dir assets/gemma-3-4b-it --epochs 20 --batch-size 128
        ```
    """

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    data_dir = args.data_dir
    model_name = args.model_name or data_dir.name
    output_dir = make_output_dir(args.output_root, model_name)

    device_str = (
        "cpu"
        if args.cpu
        else (args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    device = torch.device(device_str)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("Using device: %s", device_str)
    logging.info("Output directory: %s", output_dir)

    examples = load_examples(data_dir, args.target_layer, args.max_items)
    train_idx, val_idx, test_idx = stratified_split(examples, args.seed)

    normalizer: Normalizer | None = None
    normalizer_state: dict[str, Any] | None = None
    mean = None
    std = None
    if args.normalize == "zscore":
        logging.info("Computing z-score normalization stats on train set.")
        train_ds_raw = ActivationDataset(data_dir, examples, train_idx)
        mean, std = compute_zscore_stats(
            train_ds_raw, batch_size=args.batch_size, num_workers=args.num_workers
        )
        normalizer = ZScoreNormalizer(mean, std)
        normalizer_state = {"type": "zscore", "mean": mean, "std": std}
    elif args.normalize == "l2":
        normalizer = L2Normalizer(args.l2_eps)
        normalizer_state = {"type": "l2", "eps": float(args.l2_eps)}
    else:
        normalizer_state = {"type": "none"}

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    if top_k is not None:
        logging.info("Applying top-k=%d features per example (mode=%s).", top_k, args.top_k_mode)

    train_ds = ActivationDataset(
        data_dir,
        examples,
        train_idx,
        normalizer=normalizer,
        top_k=top_k,
        top_k_mode=args.top_k_mode,
    )
    val_ds = ActivationDataset(
        data_dir,
        examples,
        val_idx,
        normalizer=normalizer,
        top_k=top_k,
        top_k_mode=args.top_k_mode,
    )
    test_ds = ActivationDataset(
        data_dir,
        examples,
        test_idx,
        normalizer=normalizer,
        top_k=top_k,
        top_k_mode=args.top_k_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    # Determine input dimension from first batch.
    first_x, _ = next(iter(train_loader))
    input_dim = int(first_x.shape[1])
    logging.info("Input dimension: %d", input_dim)

    model = nn.Linear(input_dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_state = None
    best_val_f1 = -1.0
    history: list[dict[str, float]] = []
    patience = max(0, args.early_stop_patience)
    min_delta = max(0.0, args.early_stop_min_delta)
    no_improve_epochs = 0

    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs")
    for epoch in epoch_iter:
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            desc="Train",
            max_grad_norm=args.max_grad_norm,
        )
        val_y, val_logits = collect_logits(model, val_loader, device, desc="Val")
        val_loss = compute_bce_loss(val_y, val_logits, temperature=1.0)
        val_prob = logits_to_probs(val_logits, temperature=1.0)
        threshold = (
            args.threshold
            if args.threshold is not None
            else tune_threshold(val_y, val_prob)
        )
        val_metrics = compute_metrics(val_y, val_prob, threshold)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_f1": float(val_metrics["f1"]),
            }
        )

        if val_metrics["f1"] > best_val_f1 + min_delta:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                "state_dict": model.state_dict(),
                "threshold": threshold,
            }
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        logging.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | threshold=%.2f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_metrics["f1"],
            threshold,
        )
        epoch_iter.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
            }
        )
        if patience > 0 and no_improve_epochs >= patience:
            logging.info(
                "Early stopping at epoch %d (no val F1 improvement for %d epochs).",
                epoch,
                patience,
            )
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state["state_dict"])

    train_y, train_logits = collect_logits(
        model, train_loader, device, desc="Train Eval"
    )
    val_y, val_logits = collect_logits(model, val_loader, device, desc="Val Eval")
    test_y, test_logits = collect_logits(model, test_loader, device, desc="Test Eval")

    temperature = 1.0
    if args.temperature_scale:
        temperature = fit_temperature(val_y, val_logits)
        logging.info("Temperature scaling: T=%.4f", temperature)

    train_prob = logits_to_probs(train_logits, temperature=temperature)
    val_prob = logits_to_probs(val_logits, temperature=temperature)
    test_prob = logits_to_probs(test_logits, temperature=temperature)

    train_loss = compute_bce_loss(train_y, train_logits, temperature=temperature)
    val_loss = compute_bce_loss(val_y, val_logits, temperature=temperature)
    test_loss = compute_bce_loss(test_y, test_logits, temperature=temperature)

    if args.threshold is None:
        chosen_threshold = tune_threshold(val_y, val_prob)
    else:
        chosen_threshold = float(args.threshold)
    metrics = {
        "train": {
            **compute_metrics(train_y, train_prob, chosen_threshold),
            "loss": float(train_loss),
        },
        "val": {
            **compute_metrics(val_y, val_prob, chosen_threshold),
            "loss": float(val_loss),
        },
        "test": {
            **compute_metrics(test_y, test_prob, chosen_threshold),
            "loss": float(test_loss),
        },
        "threshold": float(chosen_threshold),
        "temperature": float(temperature),
        "history": history,
    }

    splits_payload = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "examples": [asdict(ex) for ex in examples],
    }

    normalizer_path = None
    if normalizer_state is not None and normalizer_state.get("type") == "zscore":
        normalizer_path = output_dir / "normalizer.pt"
        torch.save({"mean": mean, "std": std}, normalizer_path)
    config_payload = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "device": device_str,
        "args": vars(args),
        "input_dim": input_dim,
        "num_examples": len(examples),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "temperature": float(temperature),
        "top_k": int(args.top_k),
        "top_k_mode": args.top_k_mode,
        "normalizer": {
            "type": normalizer_state.get("type") if normalizer_state else None,
            "path": str(normalizer_path) if normalizer_path else None,
        },
    }

    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "splits.json", splits_payload)
    save_json(output_dir / "config.json", config_payload)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "threshold": float(chosen_threshold),
            "label_map": {"benign": 0, "harmful": 1},
            "temperature": float(temperature),
            "top_k": int(args.top_k),
            "top_k_mode": args.top_k_mode,
            "normalizer": normalizer_state,
        },
        output_dir / "classifier.pt",
    )

    logging.info("Saved outputs to %s", output_dir)


if __name__ == "__main__":
    main()
