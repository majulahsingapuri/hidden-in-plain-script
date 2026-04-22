"""Statistical analysis helpers for RQ2 logit-lens script composition.

The main notebook-facing workflow is:

1. ``load_rq2_results`` to read the saved RQ2 JSON output.
2. ``build_script_proportions`` to convert token ids into script proportions.
3. ``run_variant_layer_tests`` and ``run_pooled_layer_tests`` for inference.
4. ``summarize_significant_layers`` and plotting helpers for reporting.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from huggingface_hub import snapshot_download
from statsmodels.stats.multitest import multipletests
from transformers import AutoTokenizer

from utils import build_token_script_map

NON_SCRIPT_LABELS = frozenset({"Special", "Emoji", "Common", "Unknown", "Mixed"})
PRIMARY_VALUE_COLUMN = "latin_share"
SENSITIVITY_VALUE_COLUMN = "latin_share_all_tokens"


def load_rq2_results(path: str | Path) -> pd.DataFrame:
    """Load a saved RQ2 JSON result file into a DataFrame."""
    df = pd.read_json(path)
    if "prompt_id" not in df.columns:
        raise ValueError("RQ2 results must include a 'prompt_id' column.")
    df["prompt_type"] = df["prompt_id"].str.split("_").str[0]
    return df


@lru_cache(maxsize=None)
def _get_token_script_map(tokenizer_name: str) -> dict[int, str]:
    pretrained_path = snapshot_download(
        repo_id=tokenizer_name,
        local_files_only=True,
        allow_patterns=[
            "added_tokens.json",
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=True)
    return build_token_script_map(tokenizer)


def _is_named_script(label: str) -> bool:
    return label not in NON_SCRIPT_LABELS


def _script_rows_for_result(
    result: pd.Series,
    *,
    token_map: dict[int, str],
    all_scripts: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prompt_id = str(result["prompt_id"])
    prompt_type = str(result["prompt_type"])
    variant = str(result["variant"])
    for layer_idx, layer_tokens in enumerate(result["tokens"]):
        layer_labels = [token_map.get(int(tok), "Unknown") for tok in layer_tokens]
        counts = Counter(layer_labels)
        total_tokens = len(layer_labels)
        eligible_script_tokens = sum(
            count for label, count in counts.items() if _is_named_script(label)
        )
        latin_count = counts.get("Latin", 0)
        non_latin_count = sum(
            count
            for label, count in counts.items()
            if _is_named_script(label) and label != "Latin"
        )
        latin_share = (
            latin_count / eligible_script_tokens if eligible_script_tokens else np.nan
        )
        non_latin_share = (
            non_latin_count / eligible_script_tokens
            if eligible_script_tokens
            else np.nan
        )
        latin_share_all_tokens = latin_count / total_tokens if total_tokens else np.nan
        non_latin_share_all_tokens = (
            non_latin_count / total_tokens if total_tokens else np.nan
        )
        for script in all_scripts:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_type": prompt_type,
                    "variant": variant,
                    "layer": layer_idx,
                    "script": script,
                    "script_count": counts.get(script, 0),
                    "total_tokens": total_tokens,
                    "eligible_script_tokens": eligible_script_tokens,
                    "latin_count": latin_count,
                    "non_latin_count": non_latin_count,
                    "proportion": (
                        counts.get(script, 0) / total_tokens if total_tokens else 0.0
                    ),
                    "latin_share": latin_share,
                    "non_latin_share": non_latin_share,
                    "latin_share_all_tokens": latin_share_all_tokens,
                    "non_latin_share_all_tokens": non_latin_share_all_tokens,
                    "included_in_inference": eligible_script_tokens > 0,
                }
            )
    return rows


def build_script_proportions(df: pd.DataFrame, tokenizer_name: str) -> pd.DataFrame:
    """Convert token ids into per-layer script proportions and inference metrics."""
    token_map = _get_token_script_map(tokenizer_name)
    all_scripts = sorted({label for label in token_map.values()})
    rows: list[dict[str, object]] = []
    for _, result in df.iterrows():
        rows.extend(
            _script_rows_for_result(
                result,
                token_map=token_map,
                all_scripts=all_scripts,
            )
        )
    return pd.DataFrame(rows)


def _unique_layer_metrics(df_props: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "prompt_id",
        "prompt_type",
        "variant",
        "layer",
        "total_tokens",
        "eligible_script_tokens",
        "latin_count",
        "non_latin_count",
        "latin_share",
        "non_latin_share",
        "latin_share_all_tokens",
        "non_latin_share_all_tokens",
        "included_in_inference",
    ]
    return (
        df_props.loc[:, keep_cols]
        .drop_duplicates(subset=["prompt_id", "variant", "layer"])
        .sort_values(["variant", "prompt_id", "layer"], ignore_index=True)
    )


def _permutation_mean_test(
    harmful_values: Iterable[float],
    benign_values: Iterable[float],
    *,
    rng: np.random.Generator,
    n_resamples: int,
) -> tuple[float, float]:
    harmful = np.asarray(list(harmful_values), dtype=float)
    benign = np.asarray(list(benign_values), dtype=float)
    if harmful.size == 0 or benign.size == 0:
        return np.nan, np.nan

    observed = float(harmful.mean() - benign.mean())
    combined = np.concatenate([harmful, benign])
    harmful_size = harmful.size

    exceedances = 0
    observed_abs = abs(observed)
    for _ in range(n_resamples):
        shuffled = rng.permutation(combined)
        permuted = abs(shuffled[:harmful_size].mean() - shuffled[harmful_size:].mean())
        if permuted >= observed_abs:
            exceedances += 1

    p_value = (exceedances + 1) / (n_resamples + 1)
    return observed, p_value


def _apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    corrected = df.copy()
    valid = corrected["p_value"].notna()
    corrected["q_value"] = np.nan
    corrected["significant"] = False
    if valid.any():
        _, q_values, _, _ = multipletests(
            corrected.loc[valid, "p_value"],
            alpha=0.05,
            method="fdr_bh",
        )
        corrected.loc[valid, "q_value"] = q_values
        corrected.loc[valid, "significant"] = q_values <= 0.05
    return corrected


def _run_layer_tests(
    df_metrics: pd.DataFrame,
    *,
    analysis_scope: str,
    value_column: str,
    seed: int,
    n_resamples: int,
) -> pd.DataFrame:
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(df_metrics.groupby(["variant", "layer"])))
    rows: list[dict[str, object]] = []

    for child_seed, ((variant, layer), subset) in zip(
        children,
        df_metrics.groupby(["variant", "layer"], sort=True),
        strict=True,
    ):
        harmful = subset.loc[subset["prompt_type"] == "harmful", value_column].dropna()
        benign = subset.loc[subset["prompt_type"] == "benign", value_column].dropna()
        rng = np.random.default_rng(child_seed)
        mean_diff, p_value = _permutation_mean_test(
            harmful,
            benign,
            rng=rng,
            n_resamples=n_resamples,
        )
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "variant": variant,
                "layer": int(layer),
                "value_column": value_column,
                "n_harmful": int(harmful.shape[0]),
                "n_benign": int(benign.shape[0]),
                "harmful_mean": float(harmful.mean()) if not harmful.empty else np.nan,
                "benign_mean": float(benign.mean()) if not benign.empty else np.nan,
                "mean_diff": mean_diff,
                "p_value": p_value,
            }
        )

    tests = pd.DataFrame(rows).sort_values(["variant", "layer"], ignore_index=True)
    if analysis_scope == "variant":
        families = [
            _apply_fdr(family)
            for _, family in tests.groupby("variant", sort=True, dropna=False)
        ]
        return pd.concat(families, ignore_index=True)
    return _apply_fdr(tests)


def run_variant_layer_tests(
    df_props: pd.DataFrame,
    seed: int = 17,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    """Run harmful-vs-benign permutation tests for every variant and layer."""
    metrics = _unique_layer_metrics(df_props)
    eligible = metrics.loc[metrics["included_in_inference"]].copy()
    return _run_layer_tests(
        eligible,
        analysis_scope="variant",
        value_column=PRIMARY_VALUE_COLUMN,
        seed=seed,
        n_resamples=n_resamples,
    )


def run_pooled_layer_tests(
    df_props: pd.DataFrame,
    seed: int = 17,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    """Run harmful-vs-benign permutation tests after averaging across variants."""
    metrics = _unique_layer_metrics(df_props)
    eligible = metrics.loc[metrics["included_in_inference"]].copy()
    pooled = (
        eligible.groupby(["prompt_id", "prompt_type", "layer"], as_index=False)
        .agg(
            latin_share=("latin_share", "mean"),
            non_latin_share=("non_latin_share", "mean"),
            latin_share_all_tokens=("latin_share_all_tokens", "mean"),
            non_latin_share_all_tokens=("non_latin_share_all_tokens", "mean"),
            variant_count=("variant", "nunique"),
        )
        .assign(variant="ALL")
    )
    return _run_layer_tests(
        pooled,
        analysis_scope="pooled",
        value_column=PRIMARY_VALUE_COLUMN,
        seed=seed,
        n_resamples=n_resamples,
    )


def _run_sensitivity_tests(
    df_props: pd.DataFrame,
    *,
    seed: int,
    n_resamples: int,
) -> pd.DataFrame:
    metrics = _unique_layer_metrics(df_props)
    eligible = metrics.loc[metrics["included_in_inference"]].copy()
    variant_tests = _run_layer_tests(
        eligible,
        analysis_scope="variant",
        value_column=SENSITIVITY_VALUE_COLUMN,
        seed=seed,
        n_resamples=n_resamples,
    )
    pooled = (
        eligible.groupby(["prompt_id", "prompt_type", "layer"], as_index=False)
        .agg(
            latin_share_all_tokens=("latin_share_all_tokens", "mean"),
        )
        .assign(variant="ALL")
    )
    pooled_tests = _run_layer_tests(
        pooled,
        analysis_scope="pooled",
        value_column=SENSITIVITY_VALUE_COLUMN,
        seed=seed,
        n_resamples=n_resamples,
    )
    return pd.concat([variant_tests, pooled_tests], ignore_index=True)


def summarize_significant_layers(
    df_tests: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize significant layers for each analysis scope."""
    summaries: list[dict[str, object]] = []
    grouped = df_tests.groupby(["analysis_scope", "variant"], sort=True, dropna=False)
    for (analysis_scope, variant), subset in grouped:
        significant_layers = subset.loc[subset["q_value"] <= alpha, "layer"].tolist()
        summaries.append(
            {
                "analysis_scope": analysis_scope,
                "variant": variant,
                "n_significant_layers": len(significant_layers),
                "significant_layers": significant_layers,
                "min_q_value": (
                    float(subset["q_value"].min())
                    if subset["q_value"].notna().any()
                    else np.nan
                ),
                "max_abs_mean_diff": (
                    float(subset["mean_diff"].abs().max())
                    if subset["mean_diff"].notna().any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(summaries).sort_values(
        ["analysis_scope", "variant"],
        ignore_index=True,
    )


def build_sensitivity_table(
    df_props: pd.DataFrame,
    seed: int = 17,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    """Compare primary inference with an all-token denominator sensitivity run."""
    primary = pd.concat(
        [
            run_variant_layer_tests(df_props, seed=seed, n_resamples=n_resamples),
            run_pooled_layer_tests(df_props, seed=seed, n_resamples=n_resamples),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "mean_diff": "mean_diff_script_only",
            "p_value": "p_value_script_only",
            "q_value": "q_value_script_only",
            "significant": "significant_script_only",
        }
    )
    sensitivity = _run_sensitivity_tests(
        df_props,
        seed=seed,
        n_resamples=n_resamples,
    ).rename(
        columns={
            "mean_diff": "mean_diff_all_tokens",
            "p_value": "p_value_all_tokens",
            "q_value": "q_value_all_tokens",
            "significant": "significant_all_tokens",
        }
    )

    merged = primary.merge(
        sensitivity[
            [
                "analysis_scope",
                "variant",
                "layer",
                "mean_diff_all_tokens",
                "p_value_all_tokens",
                "q_value_all_tokens",
                "significant_all_tokens",
            ]
        ],
        on=["analysis_scope", "variant", "layer"],
        how="left",
    )
    merged["changed_significance"] = (
        merged["significant_script_only"] != merged["significant_all_tokens"]
    )
    return merged.sort_values(
        ["analysis_scope", "variant", "layer"],
        ignore_index=True,
    )


def plot_script_proportions(
    df_props: pd.DataFrame,
    groupby: list[str] | None = None,
    *,
    facet_row: str | None = None,
    facet_col: str | None = None,
    height: int = 800,
):
    """Replicate the notebook's descriptive script-proportion line chart."""
    if groupby is None:
        groupby = ["prompt_type", "layer", "variant", "script"]
    df_mean = df_props.groupby(groupby, as_index=False).agg(
        proportion=("proportion", "mean")
    )
    fig = px.line(
        df_mean,
        x="layer",
        y="proportion",
        color="script",
        facet_row=facet_row,
        facet_col=facet_col,
        markers=True,
        title="Per-Prompt Average Script Proportion by Layer",
        labels={
            "layer": "Layer",
            "proportion": "Proportion",
            "script": "Script",
            "prompt_type": "Prompt Type",
            "variant": "Variant",
        },
        height=height,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_mean_differences(df_tests: pd.DataFrame, *, height: int = 800):
    """Plot harmful minus benign mean differences with significant layers marked."""
    plot_df = df_tests.copy()
    plot_df["significance_label"] = np.where(
        plot_df["significant"],
        "q <= 0.05",
        "ns",
    )
    facet_row = "variant" if plot_df["analysis_scope"].eq("variant").all() else None
    fig = px.line(
        plot_df,
        x="layer",
        y="mean_diff",
        color="significance_label",
        facet_row=facet_row,
        markers=True,
        title="Harmful - Benign Latin Share by Layer",
        labels={
            "layer": "Layer",
            "mean_diff": "Mean Difference",
            "significance_label": "Significance",
            "variant": "Variant",
        },
        height=height,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    return fig


def plot_variant_heatmap(
    df_variant_tests: pd.DataFrame,
    *,
    value: str = "mean_diff",
    height: int = 900,
):
    """Plot a variant-by-layer heatmap for effect size or significance strength."""
    plot_df = df_variant_tests.copy()
    color_kwargs: dict[str, float] = {}
    color_label = value
    title_suffix = value
    if value == "-log10_q":
        safe_q = plot_df["q_value"].clip(lower=np.finfo(float).tiny)
        plot_df["-log10_q"] = -np.log10(safe_q)
        global_safe_q = df_variant_tests["q_value"].clip(lower=np.finfo(float).tiny)
        global_max = (
            float(np.nanmax(-np.log10(global_safe_q))) if len(global_safe_q) else 0.0
        )
        color_kwargs["zmin"] = 0.0
        color_kwargs["zmax"] = max(1.0, float(np.ceil(global_max * 10) / 10))
        color_label = "-log10(q-value)"
        title_suffix = "-log10(q-value)"
        value = "-log10_q"
    elif value == "mean_diff":
        global_max = (
            float(df_variant_tests["mean_diff"].abs().max())
            if df_variant_tests["mean_diff"].notna().any()
            else 0.0
        )
        bound = max(1e-9, float(np.ceil(global_max * 10) / 10))
        color_kwargs.update(
            {"zmin": -bound, "zmax": bound, "color_continuous_midpoint": 0.0}
        )
        color_label = "Mean difference"
    heatmap = plot_df.pivot(index="variant", columns="layer", values=value)
    fig = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="RdBu_r" if value == "mean_diff" else "Viridis",
        labels={"x": "Layer", "y": "Variant", "color": color_label},
        title=f"Variant by Layer Heatmap: {title_suffix}",
        height=height,
        **color_kwargs,
    )
    return fig


def format_findings_text(
    pooled_summary: pd.DataFrame,
    variant_summary: pd.DataFrame,
) -> str:
    """Build a concise notebook conclusion from summary tables."""
    pooled_row = pooled_summary.loc[pooled_summary["variant"] == "ALL"]
    if pooled_row.empty:
        pooled_text = "No pooled analysis results available."
    else:
        pooled_layers = pooled_row.iloc[0]["significant_layers"]
        pooled_text = "Pooled across variants, significant layers after FDR: " + (
            ", ".join(map(str, pooled_layers)) if pooled_layers else "none"
        )

    variant_hits = variant_summary.loc[variant_summary["n_significant_layers"] > 0]
    if variant_hits.empty:
        variant_text = "No individual variant has FDR-significant layers."
    else:
        variant_text = "Variant-specific significant layers: " + "; ".join(
            f"{row.variant}: {', '.join(map(str, row.significant_layers))}"
            for row in variant_hits.itertuples()
        )
    return pooled_text + "\n" + variant_text
