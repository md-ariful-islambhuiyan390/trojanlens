#!/usr/bin/env python3
"""Step 9 — evaluation metrics.

Paper concept
-------------
TrojanLens is scored on three axes:

  * Detection quality      — precision / recall / F1 of the module-level call.
  * Localization quality   — line-level macro-F1, precise-localization coverage
    (PLC), and IoU of the localized vs. ground-truth Trojan lines.
  * Explanation trust      — verification rate (VR): the fraction of positive
    predictions whose explanation passed the Step-8 faithfulness checks.

All metrics are computed from a saved predictions file; nothing here is
hard-coded. Each prediction record is expected to look like::

    {"file": str, "y_true": 0/1, "y_pred": 0/1, "n_lines": int,
     "gt_lines": [int], "pred_lines": [int], "verified": bool}

Pure-Python implementations (sklearn optional) so it runs offline.
"""
import argparse
import json
import os


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def detection_metrics(records):
    tp = fp = fn = tn = 0
    for r in records:
        yt, yp = int(r["y_true"]), int(r["y_pred"])
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tn += 1
    p, r_, f1 = _prf(tp, fp, fn)
    acc = (tp + tn) / len(records) if records else 0.0
    return {"precision": p, "recall": r_, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# --------------------------------------------------------------------------- #
# Localization
# --------------------------------------------------------------------------- #
def line_level_macro_f1(records):
    """Treat every (design, source-line) as a binary Trojan/clean example and
    compute macro-F1 over the two classes."""
    tp = fp = fn = tn = 0
    for r in records:
        n = int(r.get("n_lines", 0))
        gt = set(int(x) for x in r.get("gt_lines", []))
        pred = set(int(x) for x in r.get("pred_lines", []))
        universe = set(range(1, n + 1)) | gt | pred
        for ln in universe:
            g = ln in gt
            p = ln in pred
            if g and p:
                tp += 1
            elif p and not g:
                fp += 1
            elif g and not p:
                fn += 1
            else:
                tn += 1
    # positive (Trojan-line) F1
    _, _, f1_pos = _prf(tp, fp, fn)
    # negative (clean-line) F1: swap roles
    _, _, f1_neg = _prf(tn, fn, fp)
    return {"macro_f1": (f1_pos + f1_neg) / 2.0,
            "f1_trojan_line": f1_pos, "f1_clean_line": f1_neg}


def precise_localization_coverage(records, threshold: float = 0.5):
    """PLC: fraction of *trojan* designs whose predicted lines cover the
    ground-truth Trojan lines at recall >= ``threshold``."""
    trojan = [r for r in records if int(r["y_true"]) == 1]
    if not trojan:
        return 0.0
    covered = 0
    for r in trojan:
        gt = set(int(x) for x in r.get("gt_lines", []))
        pred = set(int(x) for x in r.get("pred_lines", []))
        if not gt:
            continue
        recall = len(gt & pred) / len(gt)
        if recall >= threshold:
            covered += 1
    return covered / len(trojan)


def localization_iou(records):
    """Mean IoU of predicted vs ground-truth Trojan lines over trojan designs."""
    trojan = [r for r in records if int(r["y_true"]) == 1]
    if not trojan:
        return 0.0
    ious = []
    for r in trojan:
        gt = set(int(x) for x in r.get("gt_lines", []))
        pred = set(int(x) for x in r.get("pred_lines", []))
        union = gt | pred
        ious.append((len(gt & pred) / len(union)) if union else 0.0)
    return sum(ious) / len(ious)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verification_rate(records):
    """VR: among positive PREDICTIONS, fraction with verified == True."""
    pos = [r for r in records if int(r["y_pred"]) == 1]
    if not pos:
        return 0.0
    return sum(1 for r in pos if bool(r.get("verified", False))) / len(pos)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def compute_all(records, plc_threshold: float = 0.5):
    det = detection_metrics(records)
    loc = line_level_macro_f1(records)
    return {
        "detection": det,
        "line_macro_f1": loc["macro_f1"],
        "line_f1_trojan": loc["f1_trojan_line"],
        "PLC": precise_localization_coverage(records, plc_threshold),
        "localization_IoU": localization_iou(records),
        "verification_rate": verification_rate(records),
        "n": len(records),
    }


def print_table(metrics: dict):
    d = metrics["detection"]
    rows = [
        ("designs evaluated", metrics["n"]),
        ("detection precision", f"{d['precision']:.3f}"),
        ("detection recall", f"{d['recall']:.3f}"),
        ("detection F1", f"{d['f1']:.3f}"),
        ("detection accuracy", f"{d['accuracy']:.3f}"),
        ("line-level macro-F1", f"{metrics['line_macro_f1']:.3f}"),
        ("line-level F1 (trojan)", f"{metrics['line_f1_trojan']:.3f}"),
        ("PLC (coverage)", f"{metrics['PLC']:.3f}"),
        ("localization IoU", f"{metrics['localization_IoU']:.3f}"),
        ("verification rate (VR)", f"{metrics['verification_rate']:.3f}"),
    ]
    width = max(len(k) for k, _ in rows)
    print("=" * (width + 14))
    print("TrojanLens metrics")
    print("=" * (width + 14))
    for k, v in rows:
        print(f"{k.ljust(width)} : {v}")
    print("-" * (width + 14))
    print("NOTE: computed from your predictions file only. No baked-in results.")


def read_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compute TrojanLens metrics")
    ap.add_argument("--pred", default="runs/predictions.jsonl",
                    help="predictions JSONL (produced by run_pipeline.py)")
    ap.add_argument("--plc-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    if not os.path.exists(args.pred):
        print(f"[metrics] predictions file not found: {args.pred}")
        print("[metrics] run:  python scripts/run_pipeline.py --config config.yaml "
              "--dry-run")
        return 1

    records = read_jsonl(args.pred)
    metrics = compute_all(records, plc_threshold=args.plc_threshold)
    print_table(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
