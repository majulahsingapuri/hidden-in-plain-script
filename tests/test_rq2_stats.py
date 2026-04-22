from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from rq2_stats import (
    build_script_proportions,
    plot_variant_heatmap,
    run_pooled_layer_tests,
    run_variant_layer_tests,
)


class BuildScriptProportionsTests(unittest.TestCase):
    @patch("rq2_stats._get_token_script_map")
    def test_script_bucket_logic_excludes_non_script_labels(self, mock_token_map):
        mock_token_map.return_value = {
            1: "Latin",
            2: "Gujarati",
            3: "Common",
            4: "Emoji",
            5: "Mixed",
            6: "Unknown",
            7: "Special",
        }
        df = pd.DataFrame(
            [
                {
                    "prompt_id": "harmful_0",
                    "prompt_type": "harmful",
                    "variant": "en",
                    "tokens": [[[1, 2, 3, 4, 5, 6, 7]][0]],
                }
            ]
        )

        result = build_script_proportions(df, tokenizer_name="unused")
        layer_row = result.loc[
            (result["script"] == "Latin")
            & (result["prompt_id"] == "harmful_0")
            & (result["layer"] == 0)
        ].iloc[0]

        self.assertEqual(layer_row["eligible_script_tokens"], 2)
        self.assertEqual(layer_row["latin_count"], 1)
        self.assertEqual(layer_row["non_latin_count"], 1)
        self.assertAlmostEqual(layer_row["latin_share"], 0.5)
        self.assertAlmostEqual(layer_row["non_latin_share"], 0.5)
        self.assertAlmostEqual(
            layer_row["latin_share"] + layer_row["non_latin_share"],
            1.0,
        )
        self.assertAlmostEqual(layer_row["latin_share_all_tokens"], 1 / 7)


def make_metrics_df() -> pd.DataFrame:
    variants = [
        "en",
        "gu",
        "hi",
        "ta",
        "te",
        "en_gu",
        "en_hi",
        "en_ta",
        "en_te",
        "gu_en",
        "hi_en",
        "ta_en",
        "te_en",
    ]
    rows: list[dict[str, object]] = []
    for prompt_type, offset in [("harmful", 0.25), ("benign", 0.0)]:
        for prompt_idx in range(2):
            for variant_idx, variant in enumerate(variants):
                for layer in range(34):
                    latin_share = min(1.0, offset + 0.01 * layer + 0.001 * variant_idx)
                    rows.append(
                        {
                            "prompt_id": f"{prompt_type}_{prompt_idx}",
                            "prompt_type": prompt_type,
                            "variant": variant,
                            "layer": layer,
                            "script": "Latin",
                            "total_tokens": 10,
                            "eligible_script_tokens": 10,
                            "latin_count": 6,
                            "non_latin_count": 4,
                            "proportion": latin_share,
                            "latin_share": latin_share,
                            "non_latin_share": 1 - latin_share,
                            "latin_share_all_tokens": latin_share,
                            "non_latin_share_all_tokens": 1 - latin_share,
                            "included_in_inference": True,
                        }
                    )
    return pd.DataFrame(rows)


class InferenceTests(unittest.TestCase):
    def test_variant_tests_return_expected_shape(self):
        df_props = make_metrics_df()
        result = run_variant_layer_tests(df_props, seed=17, n_resamples=499)
        self.assertEqual(result.shape[0], 13 * 34)
        self.assertEqual(result["variant"].nunique(), 13)
        for _, family in result.groupby("variant"):
            self.assertEqual(family.shape[0], 34)
            self.assertTrue(family["q_value"].notna().all())

    def test_pooled_tests_return_expected_shape(self):
        df_props = make_metrics_df()
        result = run_pooled_layer_tests(df_props, seed=17, n_resamples=499)
        self.assertEqual(result.shape[0], 34)
        self.assertTrue((result["variant"] == "ALL").all())
        self.assertTrue(result["q_value"].notna().all())
        self.assertTrue((result["n_harmful"] == 2).all())
        self.assertTrue((result["n_benign"] == 2).all())

    def test_variant_heatmap_uses_shared_symmetric_mean_diff_scale(self):
        tests_df = pd.DataFrame(
            [
                {"variant": "en", "layer": 0, "mean_diff": -0.2, "q_value": 0.4},
                {"variant": "hi", "layer": 0, "mean_diff": 0.5, "q_value": 0.02},
                {"variant": "ta", "layer": 0, "mean_diff": 1.2, "q_value": 0.8},
            ]
        )
        fig = plot_variant_heatmap(tests_df, value="mean_diff", height=300)
        self.assertEqual(fig.layout.coloraxis.cmin, -1.2)
        self.assertEqual(fig.layout.coloraxis.cmax, 1.2)
        self.assertEqual(fig.layout.coloraxis.cmid, 0.0)

    def test_variant_heatmap_uses_shared_significance_scale(self):
        tests_df = pd.DataFrame(
            [
                {"variant": "en", "layer": 0, "mean_diff": -0.2, "q_value": 0.4},
                {"variant": "hi", "layer": 0, "mean_diff": 0.5, "q_value": 0.02},
                {"variant": "ta", "layer": 0, "mean_diff": 1.2, "q_value": 0.001},
            ]
        )
        fig = plot_variant_heatmap(tests_df, value="-log10_q", height=300)
        self.assertEqual(fig.layout.coloraxis.cmin, 0.0)
        self.assertEqual(fig.layout.coloraxis.cmax, 3.0)
        self.assertIn("q=0.01", fig.layout.title.text)


if __name__ == "__main__":
    unittest.main()
