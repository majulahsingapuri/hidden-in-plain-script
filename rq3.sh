#!/usr/bin/env bash
set -euo pipefail

python rq3.py \
  --data assets/transliterations.json \
  --output assets/sae_features.json \
  --model google/gemma-3-4b-it \
  --limit 0 \
  --layers-path model.language_model.layers \
  --norm-path model.language_model.norm \
  --layer 17 \
  --sae-release gemma-scope-2-4b-it-res-all \
  --sae-id layer_17_width_262k_l0_small \
  --langs gu hi te ta
