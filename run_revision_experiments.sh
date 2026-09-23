#!/usr/bin/env bash
# ============================================================================
# TrojanLens Array R&R — CPU-side revision experiments (Apple M1 / any CPU).
# Runs: family-disjoint (M1/M3), alpha ablation (R2-5), multi-seed (m2).
# The fine-tuned obfuscation (M4) and fine-tuned multi-seed run in Colab (GPU) —
# see REVISION_EXPERIMENTS.md.
# Run from the artifact root:  bash run_revision_experiments.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs

echo "############ [M1/M3] Family-disjoint generalization ############"
python scripts/12_family_disjoint.py --config config.yaml \
    --train-family AES --test-family RS232 --out runs/fd_aes2rs232.jsonl
python scripts/12_family_disjoint.py --config config.yaml \
    --train-family RS232 --test-family AES --out runs/fd_rs2322aes.jsonl

echo "############ [R2-5] Focal-loss alpha ablation ############"
for a in 0.50 0.60 0.75 0.90; do
  echo "---- alpha=$a ----"
  python scripts/run_experiment.py --config config.yaml --alpha "$a" \
      --out "runs/alpha_${a}.jsonl"
  python scripts/07_metrics.py --pred "runs/alpha_${a}.jsonl"
done

echo "############ [m2] Multi-seed variability (frozen-feature path) ############"
for s in 13 42 101; do
  echo "---- seed=$s ----"
  python scripts/run_experiment.py --config config.yaml --seed "$s" \
      --out "runs/seed_${s}.jsonl"
  python scripts/07_metrics.py --pred "runs/seed_${s}.jsonl"
done

echo
echo "DONE. Send these files back for the paper:"
echo "  runs/fd_aes2rs232.jsonl  runs/fd_rs2322aes.jsonl   (family-disjoint -> Table 9)"
echo "  runs/alpha_*.jsonl        (alpha ablation -> Table 10)"
echo "  runs/seed_*.jsonl         (multi-seed -> Table 12)"
echo "  + paste the printed Det F1 / PLC / VR summaries."
