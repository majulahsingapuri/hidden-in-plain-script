"""Joined RQ1↔RQ2 analysis helpers.

These helpers connect RQ1 response judgments with RQ2 layer-wise logit-lens
script features after excluding gibberish responses.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

SUMMARY_LATE_LAYER_RANGE = tuple(range(15, 25))
PRIMARY_FEATURE = "latin_share"
SECONDARY_FEATURE = "non_latin_share"
HEATMAP_COEF_CAP = 200.0


def load_rq1_results(path: str | Path) -> pd.DataFrame:
    """Load a saved RQ1 JSON result file into a DataFrame."""
    df = pd.read_json(path)
    required = {"prompt_id", "variant", "refused", "harmful", "gibberish"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"RQ1 results missing columns: {sorted(missing)}")
    df["prompt_type"] = df["prompt_id"].str.extract(r"^(harmful|benign)")[0]
    return df


def _layer_metrics_from_props(df_props: pd.DataFrame) -> pd.DataFrame:
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
        .sort_values(["prompt_id", "variant", "layer"], ignore_index=True)
    )


def build_rq1_rq2_joined_frame(
    rq1_df: pd.DataFrame,
    rq2_props_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join RQ1 judgments to RQ2 layerwise script metrics and remove gibberish."""
    rq2_metrics = _layer_metrics_from_props(rq2_props_df)
    joined = rq1_df.merge(
        rq2_metrics,
        on=["prompt_id", "variant", "prompt_type"],
        how="inner",
        validate="one_to_many",
    )
    joined = joined.loc[~joined["gibberish"]].copy()
    joined["refused"] = joined["refused"].astype(int)
    joined["harmful"] = joined["harmful"].astype(int)
    joined["gibberish"] = joined["gibberish"].astype(bool)
    return joined.sort_values(
        ["prompt_type", "variant", "prompt_id", "layer"]
    ).reset_index(drop=True)


def build_joined_qa_table(
    rq1_df: pd.DataFrame,
    joined_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize join integrity and non-gibberish counts."""
    per_prompt_variant_layers = joined_df.groupby(["prompt_id", "variant"])[
        "layer"
    ].nunique()
    rows = [
        {"metric": "rq1_rows", "value": int(len(rq1_df))},
        {"metric": "non_gibberish_rows", "value": int((~rq1_df["gibberish"]).sum())},
        {"metric": "joined_rows", "value": int(len(joined_df))},
        {
            "metric": "joined_prompt_variant_pairs",
            "value": int(joined_df.groupby(["prompt_id", "variant"]).ngroups),
        },
        {
            "metric": "joined_layers_per_pair",
            "value": sorted(per_prompt_variant_layers.unique().tolist()),
        },
        {
            "metric": "non_gibberish_by_prompt_type",
            "value": joined_df[["prompt_id", "prompt_type", "variant"]]
            .drop_duplicates()["prompt_type"]
            .value_counts()
            .to_dict(),
        },
        {
            "metric": "rows_by_variant",
            "value": joined_df[["prompt_id", "variant"]]
            .drop_duplicates()["variant"]
            .value_counts()
            .to_dict(),
        },
        {
            "metric": "duplicate_prompt_variant_layer_rows",
            "value": int(joined_df.duplicated(["prompt_id", "variant", "layer"]).sum()),
        },
    ]
    return pd.DataFrame(rows)


def build_outcome_prevalence_table(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize refusal and harmful outcome prevalence after gibberish removal."""
    dedup = joined_df[
        ["prompt_id", "variant", "prompt_type", "refused", "harmful"]
    ].drop_duplicates()
    return (
        dedup.groupby(["prompt_type", "variant"], as_index=False)
        .agg(
            n_rows=("prompt_id", "size"),
            refusal_rate=("refused", "mean"),
            harmful_rate=("harmful", "mean"),
        )
        .sort_values(["prompt_type", "variant"], ignore_index=True)
    )


def _apply_fdr_by_family(df: pd.DataFrame, family_cols: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, family in df.groupby(family_cols, sort=True, dropna=False):
        corrected = family.copy()
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
        pieces.append(corrected)
    return pd.concat(pieces, ignore_index=True)


def _safe_logit_fit(design: pd.DataFrame, outcome: pd.Series) -> tuple[float, float]:
    if outcome.nunique() < 2 or design.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model = sm.Logit(outcome, design)
            result = model.fit(disp=False)
        coef = float(result.params.iloc[1])
        p_value = float(result.pvalues.iloc[1])
        return coef, p_value
    except (PerfectSeparationError, ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def _safe_odds_ratio(coef: float) -> float:
    if pd.isna(coef):
        return np.nan
    return float(np.exp(np.clip(coef, -20, 20)))


def run_variant_outcome_layer_tests(
    df_joined: pd.DataFrame,
    outcome: str,
    prompt_type: str,
    seed: int = 17,
    feature_column: str = PRIMARY_FEATURE,
) -> pd.DataFrame:
    """Fit one logistic model per layer within each variant."""
    del seed  # deterministic model fits; retained for interface stability
    subset = df_joined.loc[
        (df_joined["prompt_type"] == prompt_type)
        & df_joined["included_in_inference"]
        & df_joined[feature_column].notna()
    ].copy()
    rows: list[dict[str, object]] = []
    for (variant, layer), frame in subset.groupby(["variant", "layer"], sort=True):
        y = frame[outcome].astype(int)
        x = sm.add_constant(frame[[feature_column]], has_constant="add")
        coef, p_value = _safe_logit_fit(x, y)
        rows.append(
            {
                "analysis_scope": "variant",
                "prompt_type": prompt_type,
                "outcome": outcome,
                "variant": variant,
                "layer": int(layer),
                "feature_column": feature_column,
                "n_rows": int(len(frame)),
                "outcome_rate": float(y.mean()),
                "coef": coef,
                "odds_ratio": _safe_odds_ratio(coef),
                "p_value": p_value,
            }
        )
    results = pd.DataFrame(rows).sort_values(["variant", "layer"], ignore_index=True)
    return _apply_fdr_by_family(
        results, ["analysis_scope", "prompt_type", "outcome", "variant"]
    )


def run_pooled_adjusted_layer_tests(
    df_joined: pd.DataFrame,
    outcome: str,
    prompt_type: str,
    feature_column: str = PRIMARY_FEATURE,
) -> pd.DataFrame:
    """Fit one pooled logistic model per layer with variant adjustment."""
    subset = df_joined.loc[
        (df_joined["prompt_type"] == prompt_type)
        & df_joined["included_in_inference"]
        & df_joined[feature_column].notna()
    ].copy()
    rows: list[dict[str, object]] = []
    for layer, frame in subset.groupby("layer", sort=True):
        y = frame[outcome].astype(int)
        if y.nunique() < 2 or frame[feature_column].nunique() < 2:
            coef = np.nan
            p_value = np.nan
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=PerfectSeparationWarning)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    result = smf.logit(
                        f"{outcome} ~ {feature_column} + C(variant)",
                        data=frame,
                    ).fit(disp=False)
                coef = float(result.params[feature_column])
                p_value = float(result.pvalues[feature_column])
            except (PerfectSeparationError, ValueError, np.linalg.LinAlgError):
                coef = np.nan
                p_value = np.nan
        rows.append(
            {
                "analysis_scope": "pooled_adjusted",
                "prompt_type": prompt_type,
                "outcome": outcome,
                "variant": "ALL",
                "layer": int(layer),
                "feature_column": feature_column,
                "n_rows": int(len(frame)),
                "outcome_rate": float(y.mean()),
                "coef": coef,
                "odds_ratio": _safe_odds_ratio(coef),
                "p_value": p_value,
            }
        )
    results = pd.DataFrame(rows).sort_values("layer", ignore_index=True)
    return _apply_fdr_by_family(results, ["analysis_scope", "prompt_type", "outcome"])


def _build_summary_feature_frame(
    df_joined: pd.DataFrame,
    feature_column: str = PRIMARY_FEATURE,
) -> pd.DataFrame:
    subset = df_joined.loc[
        df_joined["included_in_inference"] & df_joined[feature_column].notna()
    ].copy()
    group_cols = ["prompt_id", "variant", "prompt_type", "refused", "harmful"]
    summary = subset.groupby(group_cols, as_index=False).agg(
        latin_share_mean_all=(feature_column, "mean"),
        latin_share_max=(feature_column, "max"),
    )
    late = (
        subset.loc[subset["layer"].isin(SUMMARY_LATE_LAYER_RANGE)]
        .groupby(group_cols, as_index=False)
        .agg(latin_share_mean_late=(feature_column, "mean"))
    )
    argmax = (
        subset.sort_values(
            ["prompt_id", "variant", feature_column, "layer"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["prompt_id", "variant"])
        .loc[:, ["prompt_id", "variant", "layer"]]
        .rename(columns={"layer": "latin_share_argmax_layer"})
    )
    return summary.merge(late, on=group_cols, how="left").merge(
        argmax, on=["prompt_id", "variant"], how="left"
    )


def run_summary_feature_models(
    df_joined: pd.DataFrame,
    feature_column: str = PRIMARY_FEATURE,
) -> pd.DataFrame:
    """Fit prompt-type-specific summary-feature logistic models."""
    summary_df = _build_summary_feature_frame(df_joined, feature_column=feature_column)
    feature_names = [
        "latin_share_mean_all",
        "latin_share_mean_late",
        "latin_share_max",
        "latin_share_argmax_layer",
    ]
    rows: list[dict[str, object]] = []
    for prompt_type in ["harmful", "benign"]:
        frame = summary_df.loc[summary_df["prompt_type"] == prompt_type].copy()
        for outcome in ["refused", "harmful"]:
            y = frame[outcome].astype(int)
            for feature in feature_names:
                if y.nunique() < 2 or frame[feature].nunique(dropna=True) < 2:
                    coef = np.nan
                    p_value = np.nan
                else:
                    design = sm.add_constant(frame[[feature]], has_constant="add")
                    coef, p_value = _safe_logit_fit(design, y)
                rows.append(
                    {
                        "analysis_scope": "summary_feature",
                        "prompt_type": prompt_type,
                        "outcome": outcome,
                        "variant": "ALL",
                        "feature_name": feature,
                        "n_rows": int(len(frame)),
                        "outcome_rate": float(y.mean()),
                        "coef": coef,
                        "odds_ratio": _safe_odds_ratio(coef),
                        "p_value": p_value,
                    }
                )
    results = pd.DataFrame(rows).sort_values(
        ["prompt_type", "outcome", "feature_name"],
        ignore_index=True,
    )
    return _apply_fdr_by_family(results, ["analysis_scope", "prompt_type", "outcome"])


def run_prediction_baselines(
    df_joined: pd.DataFrame,
    outcomes: tuple[str, ...] = ("refused", "harmful"),
) -> pd.DataFrame:
    """Train lightweight pooled predictive baselines using all layer shares."""
    subset = df_joined.loc[
        df_joined["included_in_inference"] & df_joined[PRIMARY_FEATURE].notna()
    ].copy()
    wide = (
        subset.pivot_table(
            index=["prompt_id", "variant", "prompt_type", "refused", "harmful"],
            columns="layer",
            values=PRIMARY_FEATURE,
            aggfunc="first",
        )
        .rename(columns=lambda col: f"layer_{int(col)}")
        .reset_index()
    )
    wide = wide.dropna()
    layer_cols = [col for col in wide.columns if col.startswith("layer_")]
    variant_dummies = pd.get_dummies(wide["variant"], prefix="variant", dtype=float)
    features = pd.concat([wide[layer_cols], variant_dummies], axis=1)

    rows: list[dict[str, object]] = []
    for prompt_type in ["harmful", "benign"]:
        frame = wide.loc[wide["prompt_type"] == prompt_type].copy()
        X = features.loc[frame.index]
        for outcome in outcomes:
            y = frame[outcome].astype(int)
            class_counts = y.value_counts()
            min_class = int(class_counts.min()) if not class_counts.empty else 0
            if y.nunique() < 2 or min_class < 2:
                auroc = np.nan
                average_precision = np.nan
                n_splits = np.nan
            else:
                n_splits = min(5, min_class)
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
                model = LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                )
                proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[
                    :, 1
                ]
                auroc = float(roc_auc_score(y, proba))
                average_precision = float(average_precision_score(y, proba))
            rows.append(
                {
                    "analysis_scope": "prediction",
                    "prompt_type": prompt_type,
                    "outcome": outcome,
                    "variant": "ALL",
                    "n_rows": int(len(frame)),
                    "outcome_rate": float(y.mean()),
                    "n_features": int(X.shape[1]),
                    "n_splits": n_splits,
                    "auroc": auroc,
                    "average_precision": average_precision,
                }
            )
    return pd.DataFrame(rows).sort_values(["prompt_type", "outcome"], ignore_index=True)


def summarize_joined_significant_layers(
    df_tests: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize significant layers for joined outcome models."""
    grouped = df_tests.groupby(
        ["analysis_scope", "prompt_type", "outcome", "variant"],
        sort=True,
        dropna=False,
    )
    rows: list[dict[str, object]] = []
    for (analysis_scope, prompt_type, outcome, variant), frame in grouped:
        significant_layers = frame.loc[frame["q_value"] <= alpha, "layer"].tolist()
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "prompt_type": prompt_type,
                "outcome": outcome,
                "variant": variant,
                "n_significant_layers": len(significant_layers),
                "significant_layers": significant_layers,
                "min_q_value": (
                    float(frame["q_value"].min())
                    if frame["q_value"].notna().any()
                    else np.nan
                ),
                "max_abs_coef": (
                    float(frame["coef"].abs().max())
                    if frame["coef"].notna().any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_scope", "prompt_type", "outcome", "variant"],
        ignore_index=True,
    )


def plot_variant_outcome_heatmap(
    df_variant_tests: pd.DataFrame,
    *,
    prompt_type: str,
    outcome: str,
    value: str = "coef",
    height: int = 900,
    coef_cap: float = HEATMAP_COEF_CAP,
):
    """Plot a variant-by-layer heatmap for one prompt type and one outcome."""
    plot_df = df_variant_tests.loc[
        (df_variant_tests["prompt_type"] == prompt_type)
        & (df_variant_tests["outcome"] == outcome)
    ].copy()
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
    elif value == "coef":
        global_max = (
            float(df_variant_tests["coef"].abs().max())
            if df_variant_tests["coef"].notna().any()
            else 0.0
        )
        bound = max(1e-9, min(float(np.ceil(global_max * 10) / 10), float(coef_cap)))
        color_kwargs.update(
            {"zmin": -bound, "zmax": bound, "color_continuous_midpoint": 0.0}
        )
        color_label = "Log-odds coefficient"
    heatmap = plot_df.pivot(index="variant", columns="layer", values=value)
    fig = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="RdBu_r" if value == "coef" else "Viridis",
        labels={"x": "Layer", "y": "Variant", "color": color_label},
        title=f"{prompt_type.title()} prompts, outcome={outcome}: variant-by-layer {title_suffix}",
        height=height,
        **color_kwargs,
    )
    return fig


def plot_pooled_adjusted_coefficients(
    df_pooled_tests: pd.DataFrame,
    *,
    prompt_type: str,
    outcome: str,
    height: int = 500,
):
    """Plot pooled adjusted layer coefficients for one prompt type and outcome."""
    plot_df = df_pooled_tests.loc[
        (df_pooled_tests["prompt_type"] == prompt_type)
        & (df_pooled_tests["outcome"] == outcome)
    ].copy()
    plot_df["significance_label"] = np.where(plot_df["significant"], "q <= 0.05", "ns")
    fig = px.line(
        plot_df,
        x="layer",
        y="coef",
        color="significance_label",
        markers=True,
        title=f"{prompt_type.title()} prompts, outcome={outcome}: pooled adjusted coefficients",
        labels={
            "coef": "Log-odds coefficient",
            "layer": "Layer",
            "significance_label": "Significance",
        },
        height=height,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    return fig


def format_joined_findings_text(
    variant_summary: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> str:
    """Build a short notebook-ready narrative for the joined analysis."""
    lines: list[str] = []
    for prompt_type in ["harmful", "benign"]:
        for outcome in ["refused", "harmful"]:
            pooled_row = pooled_summary.loc[
                (pooled_summary["prompt_type"] == prompt_type)
                & (pooled_summary["outcome"] == outcome)
            ]
            if pooled_row.empty:
                lines.append(f"{prompt_type}/{outcome}: no pooled adjusted result.")
                continue
            layers = pooled_row.iloc[0]["significant_layers"]
            lines.append(
                f"{prompt_type}/{outcome}: pooled adjusted significant layers = "
                + (", ".join(map(str, layers)) if layers else "none")
            )
            top_variants = variant_summary.loc[
                (variant_summary["prompt_type"] == prompt_type)
                & (variant_summary["outcome"] == outcome)
                & (variant_summary["n_significant_layers"] > 0)
            ]
            if top_variants.empty:
                lines.append(
                    f"{prompt_type}/{outcome}: no variant-specific significant layers."
                )
            else:
                parts = [
                    f"{row.variant}: {', '.join(map(str, row.significant_layers))}"
                    for row in top_variants.itertuples()
                ]
                lines.append(
                    f"{prompt_type}/{outcome}: variant-specific layers = {'; '.join(parts)}"
                )
            pred = prediction_df.loc[
                (prediction_df["prompt_type"] == prompt_type)
                & (prediction_df["outcome"] == outcome)
            ]
            if not pred.empty:
                row = pred.iloc[0]
                lines.append(
                    f"{prompt_type}/{outcome}: prediction AUROC={row['auroc']:.3f}, AP={row['average_precision']:.3f}"
                )
    return "\n".join(lines)
