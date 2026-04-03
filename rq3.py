# -*- coding: utf-8 -*-
"""SAE Feature Decomposition — Gemma Scope 2

Decomposes hidden states at a target layer into sparse, interpretable
features using a Gemma Scope 2 SAE.
"""

import argparse
import json
from tqdm import tqdm
from pathlib import Path

from nnsight import LanguageModel
from sae_lens import SAE
import torch
from config import Config


def load_model(model_name: str = "google/gemma-3-4b-it") -> LanguageModel:
    return LanguageModel(model_name, device_map="auto", dispatch=True)


def load_sae(
    release: str = "gemma-scope-2-4b-it-res-all",
    sae_id: str = "layer_17_width_262k_l0_small",
    device: str = "cuda",
) -> tuple[SAE, dict]:
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def extract_hidden_states(
    model: LanguageModel, prompt: str, target_layer: int
) -> tuple[list[str], torch.Tensor]:
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            output = model.model.language_model.layers[target_layer].output.save()
            hidden_states = model.model.language_model.norm(output)

    return hidden_states


def decompose_features(sae: SAE, hidden_states: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feature_acts = sae.encode(hidden_states)
    return feature_acts.squeeze(0)


def sae_features(
    model: LanguageModel, prompt_text, sae: SAE, target_layer: int
) -> torch.Tensor:
    hidden_states = extract_hidden_states(model, prompt_text, target_layer)
    acts = decompose_features(sae, hidden_states)
    # Mean-pool across token positions to get a single feature vector per prompt.
    return acts.mean(dim=0)


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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def build_done_sets(
    progress: list[dict],
) -> tuple[set[tuple[str, str, int]], set[tuple[str, str]]]:
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


def save_activation(
    output_dir: Path,
    prompt_id: str,
    variant: str,
    target_layer: int,
    activations: torch.Tensor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prompt_id}-{variant}-{target_layer}.pt"
    out_path = output_dir / filename
    torch.save(activations.detach().cpu(), out_path)
    return out_path


def run_batch(
    data_path: Path,
    output_path: Path,
    sae_release: str,
    sae_id: str,
    target_layer: int = 17,
    fresh: bool = False,
    langs: list[str] = [],
):

    config = Config()
    print("Config Loaded")

    variants = ["en"] + [
        version for lang in langs for version in [lang, f"{lang}_en", f"en_{lang}"]
    ]
    if fresh and output_path.exists():
        output_path.unlink()

    progress = load_progress(output_path)
    done_with_layer, done_no_layer = build_done_sets(progress)

    with open(data_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    model = load_model(config.model)
    sae = load_sae(release=sae_release, sae_id=sae_id)

    for prompt_entry in tqdm(prompts, position=0):
        for variant in tqdm(variants, position=1, leave=False):
            prompt_id = prompt_entry["prompt_id"]
            if (prompt_id, variant, target_layer) in done_with_layer or (
                prompt_id,
                variant,
            ) in done_no_layer:
                continue

            prompt_text = prompt_entry[variant]
            activations = sae_features(model, prompt_text, sae, target_layer)

            activation_path = save_activation(
                output_path.parent / "sae_activations",
                prompt_id,
                variant,
                target_layer,
                activations,
            )

            result = {
                "prompt_id": prompt_id,
                "variant": variant,
                "target_layer": target_layer,
                "activation_path": str(activation_path),
            }

            progress.append(result)
            save_progress(output_path, progress)


def main():
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
        "--layer",
        type=int,
        default=17,
        help="Target layer for hidden state extraction.",
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
    args = parser.parse_args()
    run_batch(
        args.data,
        args.output,
        args.sae_release,
        args.sae_id,
        args.layer,
        args.fresh,
    )


if __name__ == "__main__":
    main()
