# -*- coding: utf-8 -*-
"""SAE Feature Decomposition — Gemma Scope 2

Decomposes hidden states at a target layer into sparse, interpretable
features using a Gemma Scope 2 SAE.
"""

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


def sae_lens_trace(
    model: LanguageModel, sae: SAE, prompt: str
) -> dict:
    """Like logit_lens_trace but decomposes each layer's hidden state through the SAE."""
    layers = model.model.language_model.layers
    layer_acts = []

    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            input_tokens = invoker.inputs.save()
            for layer in layers:
                hidden = layer.output[0].save()
                layer_acts.append(hidden)

    input_words = [
        model.tokenizer.decode(t) for t in input_tokens[1]["input_ids"][0]
    ]

    with torch.no_grad():
        all_acts = []
        for hidden in layer_acts:
            acts = sae.encode(hidden).squeeze(0)  # (seq_len, d_sae)
            all_acts.append(acts)

    return {
        "input_words": input_words,
        "layer_acts": all_acts,  # list of (seq_len, d_sae) per layer
    }


def print_top_features(input_words: list[str], acts: torch.Tensor, top_k: int = 5):
    for pos, tok in enumerate(input_words):
        num_active = (acts[pos] > 0).sum().item()
        top_vals, top_idxs = acts[pos].topk(top_k)
        top_str = ", ".join(
            f"F{top_idxs[k].item()}({top_vals[k].item():.2f})"
            for k in range(top_k)
            if top_vals[k] > 0
        )
        print(f"  '{tok}' — {num_active} active features — top: {top_str}")


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


def main():
    target_layer = 17
    prompt = "tu kale kya jaaye che?"

    model = load_model()
    sae, cfg_dict = load_sae()
    print(f"SAE: d_in={cfg_dict['d_in']}, d_sae={cfg_dict['d_sae']}, layer={target_layer}")

    input_words, hidden_states = extract_hidden_states(model, prompt, target_layer)
    print(f"Tokens: {input_words}")
    print(f"Hidden states shape: {hidden_states.shape}")

    acts = decompose_features(sae, hidden_states)
    print_top_features(input_words, acts)
    visualize_sae_features(input_words, acts, target_layer, output_path="sae_features.html")


if __name__ == "__main__":
    main()