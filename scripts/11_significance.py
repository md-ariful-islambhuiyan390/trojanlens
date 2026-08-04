#!/usr/bin/env python3
"""Paired significance tests: frozen probe vs. LoRA fine-tuned (30-fold LOO).

Addresses the reviewer request for paired significance tests. Runs on the two
per-design prediction dumps you already have -- no GPU, no model reload:
  * frozen probe    : runs/cv_predictions.jsonl        (from run_experiment.py)
  * fine-tuned LOO  : loo_ft_records.jsonl             (from the Colab campaign)

Reports, over the designs present in BOTH files (aligned by file id):
  (a) McNemar's exact test on detection correctness (are the two detectors'
      error patterns significantly different?);
  (b) a paired bootstrap 95% CI and p-value on the detection-F1 difference; and
  (c) McNemar's exact test on verified-explanation outcomes among positive
      predictions (is the fine-tuned VR gain significant?).

Usage
-----
    python scripts/11_significance.py \
        --frozen runs/cv_predictions.jsonl \
        --finetuned loo_ft_records.jsonl
"""
import argparse
import json
import math
import os
import random


def read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def key_of(r):
    # align on the underlying design; strip any chunk suffix like "#c1"
    return str(r["file"]).split("#")[0]


def f1_of(rows):
    tp = sum(1 for r in rows if int(r["y_true"]) == 1 and int(r["y_pred"]) == 1)
    fp = sum(1 for r in rows if int(r["y_true"]) == 0 and int(r["y_pred"]) == 1)
    fn = sum(1 for r in rows if int(r["y_true"]) == 1 and int(r["y_pred"]) == 0)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c.
    p = 2 * sum_{k=0}^{min(b,c)} C(n,k) 0.5^n, clipped at 1.0."""
    n = b + c
    if n == 0:
        return 1.0, "no discordant pairs"
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail), f"b={b}, c={c}, n={n}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paired significance tests")
    ap.add_argument("--frozen", default="runs/cv_predictions.jsonl")
    ap.add_argument("--finetuned", default="loo_ft_records.jsonl")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    for p in (args.frozen, args.finetuned):
        if not os.path.exists(p):
            print(f"[sig] file not found: {p}")
            return 1

    fr = {key_of(r): r for r in read_jsonl(args.frozen)}
    ft = {key_of(r): r for r in read_jsonl(args.finetuned)}
    keys = sorted(set(fr) & set(ft))
    print(f"[sig] frozen={len(fr)} finetuned={len(ft)} aligned={len(keys)} designs")
    if not keys:
        print("[sig] no overlap between the two files (check --frozen/--finetuned).")
        return 1

    F = [fr[k] for k in keys]
    T = [ft[k] for k in keys]

    # ---- (a) McNemar on detection correctness ----------------------------- #
    b = c = 0  # b: frozen right & ft wrong ; c: frozen wrong & ft right
    for f, t in zip(F, T):
        fc = int(f["y_pred"]) == int(f["y_true"])
        tc = int(t["y_pred"]) == int(t["y_true"])
        if fc and not tc:
            b += 1
        elif tc and not fc:
            c += 1
    p_det, det_info = mcnemar_exact(b, c)
    print("\n=== (a) Detection correctness: McNemar's exact test ===")
    print(f"    frozen-right/ft-wrong b={b}; frozen-wrong/ft-right c={c}")
    print(f"    p = {p_det:.4f}  ({'significant' if p_det < 0.05 else 'not significant'} at 0.05)")

    # ---- (b) Paired bootstrap on detection F1 ----------------------------- #
    rng = random.Random(args.seed)
    n = len(keys)
    f1_fr, f1_ft = f1_of(F), f1_of(T)
    diffs = []
    for _ in range(args.boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(f1_of([T[i] for i in idx]) - f1_of([F[i] for i in idx]))
    diffs.sort()
    lo = diffs[int(0.025 * args.boot)]
    hi = diffs[int(0.975 * args.boot)]
    # two-sided bootstrap p: fraction of resamples on the other side of 0, x2
    p_boot = 2.0 * min(sum(1 for d in diffs if d <= 0),
                       sum(1 for d in diffs if d >= 0)) / args.boot
    p_boot = min(1.0, p_boot)
    print("\n=== (b) Detection F1 difference (fine-tuned - frozen): paired bootstrap ===")
    print(f"    F1 frozen={f1_fr:.3f}  fine-tuned={f1_ft:.3f}  delta={f1_ft-f1_fr:+.3f}")
    print(f"    95% CI [{lo:+.3f}, {hi:+.3f}]   bootstrap p = {p_boot:.4f}")

    # ---- (c) McNemar on verified-explanation outcomes (positive preds) ---- #
    # among designs predicted positive by BOTH, did verification outcomes differ?
    vb = vc = 0
    both_pos = 0
    for f, t in zip(F, T):
        if int(f["y_pred"]) == 1 and int(t["y_pred"]) == 1:
            both_pos += 1
            fv = bool(f.get("verified"))
            tv = bool(t.get("verified"))
            if fv and not tv:
                vb += 1
            elif tv and not fv:
                vc += 1
    p_vr, vr_info = mcnemar_exact(vb, vc)
    print("\n=== (c) Verified explanations among shared positive predictions: McNemar ===")
    print(f"    shared positive predictions={both_pos}")
    print(f"    frozen-verified/ft-not vb={vb}; frozen-not/ft-verified vc={vc}")
    print(f"    p = {p_vr:.4f}  ({'significant' if p_vr < 0.05 else 'not significant'} at 0.05)")

    print("\n=== PAPER-READY sentence ===")
    vr_sig = ("a significant increase" if p_vr < 0.05
              else "an increase (not significant at the 0.05 level)")
    det_sig = "significant" if p_boot < 0.05 else "not significant"
    print(f"Under 30-fold LOO, the fine-tuned model's verified-explanation rate "
          f"showed {vr_sig} over the frozen probe (McNemar p={p_vr:.3g}); the change "
          f"in detection F1 was {f1_ft-f1_fr:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}], "
          f"bootstrap p={p_boot:.3g}, {det_sig}).")
    print("NOTE: computed from your prediction files only; no baked-in results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
