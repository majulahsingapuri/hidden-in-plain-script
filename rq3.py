# -*- coding: utf-8 -*-
"""SAE Feature Decomposition — Gemma Scope 2

Decomposes hidden states at a target layer into sparse, interpretable
features using a Gemma Scope 2 SAE.
"""

import argparse
import json
from tqdm import tqdm
from pathlib import Path
from typing import Any

from nnsight import LanguageModel
from sae_lens import SAE
import torch
from utils import resolve_attr_path, ResourceMonitor


def load_model(model_name: str = "google/gemma-3-4b-it") -> LanguageModel:
    """Load the traced language model used for SAE feature extraction.

    Example:
        >>> # model = load_model("google/gemma-3-4b-it")
    """

    return LanguageModel(model_name, device_map="auto", dispatch=True)


def load_sae(
    release: str = "gemma-scope-2-4b-it-res-all",
    sae_id: str = "layer_17_width_262k_l0_small",
    device: str = "cuda",
) -> tuple[SAE, dict]:
    """Load a pretrained SAE from SAE Lens.

    Args:
        release: SAE release collection name.
        sae_id: Specific SAE identifier inside the release.
        device: Device string understood by SAE Lens.
    """

    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def get_tokenizer(model: LanguageModel) -> Any:
    """Return the tokenizer attached to a traced language model."""

    tokenizer = getattr(model, "tokenizer", None) or getattr(model, "_tokenizer", None)
    if tokenizer is None:
        raise AttributeError("LanguageModel does not expose a tokenizer.")
    return tokenizer


def tokenize_prompt(model: LanguageModel, prompt: str) -> tuple[list[int], list[str]]:
    """Tokenize a prompt into token IDs and decoded token strings."""

    tokenizer = get_tokenizer(model)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    token_ids_tensor = encoded["input_ids"][0].detach().cpu()
    token_ids = [int(token_id) for token_id in token_ids_tensor.tolist()]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(token_ids)]
    else:
        tokens = [str(tokenizer.decode([token_id])) for token_id in token_ids]
    return token_ids, tokens


def extract_hidden_states(
    model: LanguageModel,
    prompt: str,
    target_layer: int,
    layers_path: str,
    norm_path: str,
) -> torch.Tensor:
    """Trace one prompt and return hidden states from the selected layer."""

    layers = resolve_attr_path(model, layers_path)
    norm = resolve_attr_path(model, norm_path)
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            output = layers[target_layer].output.save()
            hidden_states = norm(output)
            hidden_states.save()

    return hidden_states


def decompose_features(sae: SAE, hidden_states: torch.Tensor) -> torch.Tensor:
    """Encode hidden states into sparse SAE feature activations."""

    with torch.no_grad():
        feature_acts = sae.encode(hidden_states)
    return feature_acts.squeeze(0)


def sae_token_features(
    model: LanguageModel,
    prompt_text,
    sae: SAE,
    target_layer: int,
    layers_path: str,
    norm_path: str,
) -> torch.Tensor:
    """Return per-token SAE feature activations for a prompt."""

    hidden_states = extract_hidden_states(
        model,
        prompt_text,
        target_layer,
        layers_path,
        norm_path,
    )
    return decompose_features(sae, hidden_states)


def sae_features(token_acts: torch.Tensor) -> torch.Tensor:
    """Mean-pool per-token SAE activations into one vector per prompt."""

    return token_acts.mean(dim=0)


def load_progress(output_path: Path) -> list[dict]:
    """Return list of processed entries stored in a JSON file."""
    if not output_path.exists():
        return []

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []

    data = json.loads(content)

    if isinstance(data, dict) and "processed" in data:
        data = data["processed"]
    return data if isinstance(data, list) else []


def save_progress(output_path: Path, progress: list[dict]):
    """Persist incremental RQ3 progress to disk as JSON."""

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def build_done_sets(
    progress: list[dict],
) -> tuple[set[tuple[str, str, int]], set[tuple[str, str]]]:
    """Build lookup sets for already-processed prompt variants."""

    done_with_layer = set()
    done_no_layer = set()
    for entry in progress:
        prompt_id = entry.get("prompt_id")
        variant = entry.get("variant")
        if not prompt_id or not variant:
            continue
        layer = entry.get("target_layer")
        if layer is None:
            done_no_layer.add((prompt_id, variant))
        else:
            done_with_layer.add((prompt_id, variant, layer))
    return done_with_layer, done_no_layer


def build_progress_index(
    progress: list[dict],
) -> tuple[dict[tuple[str, str, int], int], dict[tuple[str, str], int]]:
    """Build index maps for progress rows keyed by prompt variant."""

    index_with_layer = {}
    index_no_layer = {}
    for idx, entry in enumerate(progress):
        prompt_id = entry.get("prompt_id")
        variant = entry.get("variant")
        if not prompt_id or not variant:
            continue
        layer = entry.get("target_layer")
        if layer is None:
            index_no_layer[(prompt_id, variant)] = idx
        else:
            index_with_layer[(prompt_id, variant, int(layer))] = idx
    return index_with_layer, index_no_layer


def save_activation(
    output_dir: Path,
    prompt_id: str,
    variant: str,
    target_layer: int,
    activations: torch.Tensor,
) -> Path:
    """Save one activation vector and return the created path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prompt_id}-{variant}-{target_layer}.pt"
    out_path = output_dir / filename
    torch.save(activations.detach().cpu(), out_path)
    return out_path


def sparsify_token_activations(
    token_acts: torch.Tensor, token_top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the top-k latents for each token position."""

    if token_acts.dim() != 2:
        raise ValueError(f"Expected token activations with shape [T, D], got {tuple(token_acts.shape)}")
    k = max(1, min(int(token_top_k), int(token_acts.shape[1])))
    values, indices = torch.topk(token_acts, k=k, dim=1)
    return indices.detach().cpu(), values.detach().cpu()


def save_token_activation(
    output_dir: Path,
    prompt_id: str,
    variant: str,
    target_layer: int,
    prompt_text: str,
    token_ids: list[int],
    tokens: list[str],
    token_acts: torch.Tensor,
    token_top_k: int,
) -> Path:
    """Save sparse per-token latent activations and return the created path."""

    if token_acts.dim() != 2:
        raise ValueError(f"Expected token activations with shape [T, D], got {tuple(token_acts.shape)}")

    seq_len = min(len(token_ids), len(tokens), int(token_acts.shape[0]))
    token_ids = token_ids[:seq_len]
    tokens = tokens[:seq_len]
    token_acts = token_acts[:seq_len]
    top_indices, top_values = sparsify_token_activations(token_acts, token_top_k)

    payload = {
        "prompt_id": prompt_id,
        "variant": variant,
        "target_layer": int(target_layer),
        "prompt_text": prompt_text,
        "token_ids": token_ids,
        "tokens": tokens,
        "top_latent_indices": top_indices,
        "top_latent_values": top_values,
        "token_top_k": int(top_indices.shape[1]),
        "seq_len": int(seq_len),
        "num_latents": int(token_acts.shape[1]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prompt_id}-{variant}-{target_layer}.pt"
    out_path = output_dir / filename
    torch.save(payload, out_path)
    return out_path


def run_batch(
    data_path: Path,
    output_path: Path,
    model_name: str,
    sae_release: str,
    sae_id: str,
    limit: int = 0,
    target_layer: int = 17,
    layers_path: str = "model.language_model.layers",
    norm_path: str = "model.language_model.norm",
    fresh: bool = False,
    langs: list[str] = [],
    save_token_activations: bool = False,
    token_top_k: int = 256,
    token_output_dir: Path | None = None,
):
    """Run SAE feature extraction over every requested prompt variant.

    Args:
        data_path: Dataset containing transliterated prompt variants.
        output_path: JSON index that records saved activation files.
        model_name: Hugging Face model name or local path.
        sae_release: SAE release collection name.
        sae_id: SAE identifier inside the release.
        limit: Optional prompt limit. `0` means no limit.
        target_layer: Transformer layer to trace before SAE encoding.
        layers_path: Dotted path to the transformer layer list.
        norm_path: Dotted path to the final norm module.
        fresh: Delete existing progress JSON before starting.
        langs: Target language codes used to construct prompt variants.
        save_token_activations: Persist sparse token-level SAE activations.
        token_top_k: Keep the top-k latents per token position.
        token_output_dir: Optional output directory for token-level artifacts.

    Example:
        >>> # run_batch(Path("assets/transliterations.json"), Path("assets/gemma-3-4b-it/sae_features.json"), "google/gemma-3-4b-it", "gemma-scope-2-4b-it-res-all", "layer_17_width_262k_l0_small")
    """

    variants = ["en"] + [
        version for lang in langs for version in [lang, f"{lang}_en", f"en_{lang}"]
    ]
    if fresh and output_path.exists():
        output_path.unlink()

    progress = load_progress(output_path)
    done_with_layer, done_no_layer = build_done_sets(progress)
    progress_index_with_layer, progress_index_no_layer = build_progress_index(progress)

    with open(data_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if limit:
        prompts = prompts[:limit]

    model = load_model(model_name)
    sae = load_sae(release=sae_release, sae_id=sae_id)

    monitor = ResourceMonitor()
    prompts_bar = tqdm(prompts, position=0)
    for prompt_entry in prompts_bar:
        variants_bar = tqdm(variants, position=1, leave=False)
        for variant in variants_bar:
            postfix = monitor.tqdm_postfix()
            if postfix:
                variants_bar.set_postfix(postfix, refresh=False)
            prompt_id = prompt_entry["prompt_id"]
            existing_idx = progress_index_with_layer.get(
                (prompt_id, variant, target_layer)
            )
            if existing_idx is None:
                existing_idx = progress_index_no_layer.get((prompt_id, variant))
            existing_entry = progress[existing_idx] if existing_idx is not None else None

            has_existing_activation = bool(existing_entry and existing_entry.get("activation_path"))
            has_existing_token_activation = bool(
                existing_entry and existing_entry.get("token_activation_path")
            )

            if has_existing_activation and (
                not save_token_activations or has_existing_token_activation
            ):
                continue

            prompt_text = prompt_entry[variant]
            token_acts = sae_token_features(
                model,
                prompt_text,
                sae,
                target_layer,
                layers_path,
                norm_path,
            )
            activations = sae_features(token_acts)

            activation_path = save_activation(
                output_path.parent / "sae_activations",
                prompt_id,
                variant,
                target_layer,
                activations,
            )

            result = dict(existing_entry) if existing_entry else {}
            token_activation_path = None
            if save_token_activations:
                token_ids, tokens = tokenize_prompt(model, prompt_text)
                token_activation_path = save_token_activation(
                    token_output_dir or (output_path.parent / "sae_token_activations"),
                    prompt_id,
                    variant,
                    target_layer,
                    prompt_text,
                    token_ids,
                    tokens,
                    token_acts,
                    token_top_k,
                )

            result.update(
                {
                    "prompt_id": prompt_id,
                    "variant": variant,
                    "target_layer": target_layer,
                    "activation_path": str(activation_path),
                }
            )
            if token_activation_path is not None:
                result["token_activation_path"] = str(token_activation_path)
                result["token_top_k"] = int(token_top_k)

            if existing_idx is None:
                progress.append(result)
                new_idx = len(progress) - 1
                progress_index_with_layer[(prompt_id, variant, target_layer)] = new_idx
            else:
                progress[existing_idx] = result
            save_progress(output_path, progress)

    monitor.close()


def main():
    """Parse CLI arguments and run RQ3."""

    parser = argparse.ArgumentParser(
        description="SAE feature decomposition over transliterations."
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default=Path.cwd() / "assets" / "transliterations.json",
        help="Path to transliterations JSON.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path.cwd() / "assets" / "sae_features.json",
        help="Output progress JSON file.",
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
        "--layer",
        type=int,
        default=17,
        help="Target layer for hidden state extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of data points used for run.",
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2-4b-it-res-all",
        help="SAE release name.",
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        default="layer_17_width_262k_l0_small",
        help="SAE release id",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch (delete existing output).",
    )
    parser.add_argument(
        "--langs",
        "-l",
        nargs="+",
        default=["gu", "hi", "ta", "te"],
        help="Target language codes (e.g., gu te).",
    )
    parser.add_argument(
        "--save-token-activations",
        action="store_true",
        help="Also save sparse token-level SAE activations for later highlighting.",
    )
    parser.add_argument(
        "--token-top-k",
        type=int,
        default=256,
        help="Top-k SAE latents to keep per token when saving token activations.",
    )
    parser.add_argument(
        "--token-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for sparse token-level SAE activations.",
    )
    args = parser.parse_args()
    run_batch(
        args.data,
        args.output,
        args.model,
        args.sae_release,
        args.sae_id,
        args.limit,
        args.layer,
        args.layers_path,
        args.norm_path,
        args.fresh,
        args.langs,
        args.save_token_activations,
        args.token_top_k,
        args.token_output_dir,
    )


if __name__ == "__main__":
    main()
