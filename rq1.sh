#!/usr/bin/env bash
set -euo pipefail

python rq1.py \
  --data ./assets/transliterations.json \
  --model "google/gemma-3-4b-it" \
  --limit 0 \
  --langs gu hi te ta \
  --batch-size 128 \
  --judge-concurrency 32
