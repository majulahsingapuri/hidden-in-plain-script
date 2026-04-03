#!/usr/bin/env bash
set -euo pipefail

python rq2.py \
  --data assets/transliterations.json \
  --model google/gemma-3-4b-it \
  --limit 0 \
  --layers-path model.language_model.layers \
  --norm-path model.language_model.norm \
  --langs gu hi te ta \
  --batch-size 16
