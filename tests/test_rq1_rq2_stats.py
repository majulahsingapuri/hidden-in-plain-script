from __future__ import annotations

import unittest

import pandas as pd

from rq1_rq2_stats import (
    build_outcome_prevalence_table,
    build_rq1_rq2_joined_frame,
    plot_variant_outcome_heatmap,
    run_pooled_adjusted_layer_tests,
    run_prediction_baselines,
    run_summary_feature_models,
    run_variant_outcome_layer_tests,
)


def make_rq1_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = ["en", "hi"]
    for prompt_type in ["harmful", "benign"]:
        for prompt_idx in range(4):
            for variant in variants:
                gibberish = prompt_type == "benign" and prompt_idx == 3 and variant == "hi"
                refused = int((prompt_idx + (variant == "hi")) % 2 == 0)
                harmful = int((prompt_idx + (variant == "en")) % 2 == 1)
                rows.append(
                    {
                        "prompt_id": f"{prompt_type}_{prompt_idx}",
                        "variant": variant,
                        "prompt_type": prompt_type,
                        "refused": refused,
                        "harmful": harmful,
                        "gibberish": gibberish,
                    }
                )
    return pd.DataFrame(rows)


def make_rq2_props_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = ["en", "hi"]
    for prompt_type, base in [("harmful", 0.55), ("benign", 0.35)]:
        for prompt_idx in range(4):
            for variant_idx, variant in enumerate(variants):
                for layer in range(34):
                    latin_share = min(0.99, base + 0.01 * layer + 0.02 * prompt_idx + 0.03 * variant_idx)
                    rows.append(
                        {
                            "prompt_id": f"{prompt_type}_{prompt_idx}",
                            "prompt_type": prompt_type,
                            "variant": variant,
                            "layer": layer,
                            "total_tokens": 10,
                            "eligible_script_tokens": 10,
                            "latin_count": int(round(latin_share * 10)),
                            "non_latin_count": 10 - int(round(latin_share * 10)),
                            "latin_share": latin_share,
                            "non_latin_share": 1 - latin_share,
                            "latin_share_all_tokens": latin_share,
                            "non_latin_share_all_tokens": 1 - latin_share,
                            "included_in_inference": True,
                        }
                    )
    return pd.DataFrame(rows)


class JoinedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rq1_df = make_rq1_df()
        self.rq2_props_df = make_rq2_props_df()
        self.joined = build_rq1_rq2_joined_frame(self.rq1_df, self.rq2_props_df)

    def test_join_discards_gibberish_and_keeps_layer_rows(self):
        non_gib_pairs = (~self.rq1_df["gibberish"]).sum()
        self.assertEqual(len(self.joined), non_gib_pairs * 34)
        self.assertFalse(self.joined["gibberish"].any())
        self.assertEqual(self.joined.duplicated(["prompt_id", "variant", "layer"]).sum(), 0)

    def test_outcome_prevalence_table_has_prompt_type_variant_rows(self):
        prevalence = build_outcome_prevalence_table(self.joined)
        self.assertEqual(set(prevalence.columns), {"prompt_type", "variant", "n_rows", "refusal_rate", "harmful_rate"})
        self.assertEqual(prevalence.shape[0], 4)

    def test_variant_layer_models_return_expected_shape(self):
        result = run_variant_outcome_layer_tests(self.joined, outcome="refused", prompt_type="harmful")
        self.assertEqual(result.shape[0], 2 * 34)
        self.assertEqual(result["variant"].nunique(), 2)
        self.assertTrue(result["q_value"].notna().all())

    def test_pooled_adjusted_models_return_expected_shape(self):
        result = run_pooled_adjusted_layer_tests(self.joined, outcome="harmful", prompt_type="benign")
        self.assertEqual(result.shape[0], 34)
        self.assertTrue((result["variant"] == "ALL").all())
        self.assertTrue(result["q_value"].notna().all())

    def test_summary_feature_models_cover_all_prompt_type_outcome_pairs(self):
        result = run_summary_feature_models(self.joined)
        self.assertEqual(result.shape[0], 16)
        self.assertEqual(sorted(result["prompt_type"].unique().tolist()), ["benign", "harmful"])
        self.assertEqual(sorted(result["outcome"].unique().tolist()), ["harmful", "refused"])

    def test_prediction_baselines_return_all_pairs(self):
        result = run_prediction_baselines(self.joined)
        self.assertEqual(result.shape[0], 4)
        self.assertTrue((result["n_features"] > 0).all())
        self.assertTrue((result["n_splits"] >= 2).all())

    def test_variant_heatmap_uses_shared_symmetric_coef_scale(self):
        tests_df = pd.DataFrame(
            [
                {"prompt_type": "harmful", "outcome": "refused", "variant": "en", "layer": 0, "coef": -0.3, "q_value": 0.2},
                {"prompt_type": "harmful", "outcome": "refused", "variant": "hi", "layer": 0, "coef": 0.1, "q_value": 0.03},
                {"prompt_type": "benign", "outcome": "refused", "variant": "en", "layer": 0, "coef": 1.8, "q_value": 0.5},
            ]
        )
        fig = plot_variant_outcome_heatmap(
            tests_df,
            prompt_type="harmful",
            outcome="refused",
            value="coef",
            height=300,
        )
        self.assertEqual(fig.layout.coloraxis.cmin, -1.8)
        self.assertEqual(fig.layout.coloraxis.cmax, 1.8)
        self.assertEqual(fig.layout.coloraxis.cmid, 0.0)

    def test_variant_heatmap_clips_extreme_coef_outliers_by_default(self):
        tests_df = pd.DataFrame(
            [
                {"prompt_type": "harmful", "outcome": "refused", "variant": "en", "layer": 0, "coef": -35.0, "q_value": 0.2},
                {"prompt_type": "harmful", "outcome": "refused", "variant": "hi", "layer": 0, "coef": 120.0, "q_value": 0.03},
                {"prompt_type": "benign", "outcome": "refused", "variant": "en", "layer": 0, "coef": 2400.0, "q_value": 0.5},
            ]
        )
        fig = plot_variant_outcome_heatmap(
            tests_df,
            prompt_type="harmful",
            outcome="refused",
            value="coef",
            height=300,
        )
        self.assertEqual(fig.layout.coloraxis.cmin, -200.0)
        self.assertEqual(fig.layout.coloraxis.cmax, 200.0)
        self.assertEqual(fig.layout.coloraxis.cmid, 0.0)
        self.assertIn("clipped to +/-200", fig.layout.title.text)

    def test_variant_heatmap_uses_shared_significance_scale(self):
        tests_df = pd.DataFrame(
            [
                {"prompt_type": "harmful", "outcome": "refused", "variant": "en", "layer": 0, "coef": -0.3, "q_value": 0.2},
                {"prompt_type": "harmful", "outcome": "refused", "variant": "hi", "layer": 0, "coef": 0.1, "q_value": 0.03},
                {"prompt_type": "benign", "outcome": "refused", "variant": "en", "layer": 0, "coef": 1.8, "q_value": 0.001},
            ]
        )
        fig = plot_variant_outcome_heatmap(
            tests_df,
            prompt_type="harmful",
            outcome="refused",
            value="-log10_q",
            height=300,
        )
        self.assertEqual(fig.layout.coloraxis.cmin, 0.0)
        self.assertEqual(fig.layout.coloraxis.cmax, 3.0)
        self.assertIn("q=0.05", fig.layout.title.text)


if __name__ == "__main__":
    unittest.main()
