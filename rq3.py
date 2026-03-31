# -*- coding: utf-8 -*-
"""SAE Feature Decomposition — Gemma Scope 2

Decomposes hidden states at a target layer into sparse, interpretable
features using a Gemma Scope 2 SAE.
"""

import argparse
import json
from pathlib import Path

from nnsight import LanguageModel
from sae_lens import SAE
import torch
import plotly.express as px


def load_model(model_name: str = "google/gemma-3-4b-it") -> LanguageModel:
    return LanguageModel(model_name, device_map="auto", dispatch=True)


def load_sae(
    release: str = "gemma-scope-2-4b-it-res-all",
    sae_id: str = "layer_17_width_262k_l0_small",
    device: str = "mps",
) -> tuple[SAE, dict]:
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device
    )
    return sae, cfg_dict


def extract_hidden_states(
    model: LanguageModel, prompt: str, target_layer: int
) -> tuple[list[str], torch.Tensor]:
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            input_tokens = invoker.inputs.save()
            hidden_states = (
                model.model.language_model.layers[target_layer].output[0].save()
            )

    input_words = [
        model.tokenizer.decode(t) for t in input_tokens[1]["input_ids"][0]
    ]
    return input_words, hidden_states


def decompose_features(
    sae: SAE, hidden_states: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        feature_acts = sae.encode(hidden_states)
    return feature_acts.squeeze(0)


def extract_top_features(acts: torch.Tensor, top_k: int = 5) -> list[dict]:
    results = []
    for pos in range(acts.shape[0]):
        num_active = (acts[pos] > 0).sum().item()
        top_vals, top_idxs = acts[pos].topk(top_k)
        top_features = [
            {"feature_id": top_idxs[k].item(), "activation": top_vals[k].item()}
            for k in range(top_k)
            if top_vals[k] > 0
        ]
        results.append({
            "num_active": num_active,
            "top_features": top_features,
        })
    return results



def sae_features(model: LanguageModel, prompt_text, sae: SAE, target_layer: int, top_k: int) -> tuple[
    list[str], list[dict]]:
    input_words, hidden_states = extract_hidden_states(model, prompt_text, target_layer)
    acts = decompose_features(sae, hidden_states)
    token_features = extract_top_features(acts, top_k)
    return input_words, token_features



def visualize_sae_features(
    input_words: list[str],
    acts: torch.Tensor,
    target_layer: int,
    top_n: int = 30,
    output_path: str | None = None,
):
    max_per_feature = acts.max(dim=0).values
    top_feature_ids = max_per_feature.topk(top_n).indices

    sub_acts = acts[:, top_feature_ids].T.cpu().float().numpy()
    feature_labels = [f"F{fid.item()}" for fid in top_feature_ids]

    fig = px.imshow(
        sub_acts,
        x=input_words,
        y=feature_labels,
        color_continuous_scale=px.colors.diverging.RdYlBu_r,
        color_continuous_midpoint=0.0,
        text_auto=".1f",
        labels=dict(x="Input Tokens", y="SAE Features", color="Activation"),
    )
    fig.update_layout(
        title=f"SAE Feature Activations at Layer {target_layer}",
        xaxis_tickangle=0,
        height=700,
    )
    fig.show()

    if output_path:
        fig.write_html(output_path)


def load_progress(output_path: Path) -> set[tuple[str, str]]:
    """Return set of (prompt_id, variant) already processed."""
    done = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                done.add((entry["prompt_id"], entry["variant"]))
    return done


def append_result(output_path: Path, result: dict):
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")



VARIANT_KEYS = ["en", "en_gu", "en_hi", "en_ta", "en_te",
                "gu", "gu_en", "hi", "hi_en", "ta", "ta_en", "te", "te_en"]


def run_batch(
    data_path: Path,
    output_path: Path,
    target_layer: int = 17,
    top_k: int = 5,
    fresh: bool = False,
):
    if fresh and output_path.exists():
        output_path.unlink()

    done = load_progress(output_path)
    print(f"Already processed: {len(done)} entries")

    with open(data_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    model = load_model()
    sae, cfg_dict = load_sae()
    print(f"SAE: d_in={cfg_dict['d_in']}, d_sae={cfg_dict['d_sae']}, layer={target_layer}")

    total = sum(
        1 for p in prompts for v in VARIANT_KEYS
        if v in p and (p["prompt_id"], v) not in done
    )
    processed = 0

    for prompt_entry in prompts:
        prompt_id = prompt_entry["prompt_id"]

        for variant in VARIANT_KEYS:
            if variant not in prompt_entry:
                continue
            if (prompt_id, variant) in done:
                continue

            prompt_text = prompt_entry[variant]
            input_words, token_features = sae_features(model, prompt_text, sae, target_layer, top_k)

            result = {
                "prompt_id": prompt_id,
                "variant": variant,
                "prompt_text": prompt_text,
                "target_layer": target_layer,
                "input_words": input_words,
                "token_features": token_features,
            }

            append_result(output_path, result)
            processed += 1
            print(f"[{processed}/{total}] {prompt_id} / {variant}")

    print(f"Done. Wrote {processed} new entries to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SAE feature decomposition over transliterations.")
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path.cwd() / "assets" / "transliterations.json",
        help="Path to transliterations JSON.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path.cwd() / "assets" / "sae_features.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=17,
        help="Target layer for hidden state extraction.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top features per token.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from scratch (delete existing output).",
    )
    args = parser.parse_args()
    run_batch(args.data, args.output, args.layer, args.top_k, args.fresh)


if __name__ == "__main__":
    main()