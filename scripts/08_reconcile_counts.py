#!/usr/bin/env python3
"""Phase-1 integrity check — reconcile the module/positive counts used across
the paper (data table vs. confusion matrix vs. CV predictions).

Reviewer 1 flagged (implicitly) that the numbers must agree. The pooled
confusion figure shows 89 TP + 1 FN = 90 positives, while the data table
(Table `tab:data`) says 95 Trojan-positive modules. This script finds the
source of that gap so you can fix whichever number is stale BEFORE resubmission.

It is pure-Python (only needs PyYAML) and reads:
  * the processed corpus            cfg['paths']['processed']  (combined.jsonl)
  * the CV predictions (if present) cfg['paths']['runs']/cv_predictions.jsonl

Usage
-----
    python scripts/08_reconcile_counts.py --config config.yaml

Output: a per-family census of the corpus, the same census restricted to the
designs that actually appear in cv_predictions.jsonl, and an explicit list of
any positive designs that are in the corpus but MISSING from the predictions
(the most likely cause of a 95 -> 90 drop, e.g. designs skipped by a max-length
cap during evaluation).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def variant_of(file_path):
    """Top-level benchmark folder, e.g. 'RS232-T2100/src/TjIn/u_rec.v' -> 'RS232-T2100'."""
    return file_path.replace("\\", "/").split("/")[0]


def family_of(variant):
    """'RS232-T2100' -> 'RS232'; 'AES-T1000' -> 'AES'. Splits on the first
    Trojan-id delimiter ('-' or '_') so digits in the family name (RS232) stay."""
    return re.split(r"[-_]", variant, maxsplit=1)[0].upper()


def census(records):
    """Return per-family {modules, positive, clean, variants, pos_variants}."""
    fam = {}
    for r in records:
        v = variant_of(r["file"])
        f = family_of(v)
        d = fam.setdefault(f, {"modules": 0, "positive": 0, "clean": 0,
                               "variants": set(), "pos_variants": set()})
        d["modules"] += 1
        d["variants"].add(v)
        if int(r.get("label", 0)) == 1:
            d["positive"] += 1
            d["pos_variants"].add(v)
        else:
            d["clean"] += 1
    return fam


def print_census(title, fam):
    print(f"\n{title}")
    print("-" * len(title))
    tot = {"modules": 0, "positive": 0, "clean": 0, "variants": 0, "pos_variants": 0}
    hdr = f"{'family':<8} {'variants':>8} {'pos_var':>8} {'modules':>8} {'Trojan+':>8} {'clean':>8}"
    print(hdr)
    for f in sorted(fam):
        d = fam[f]
        print(f"{f:<8} {len(d['variants']):>8} {len(d['pos_variants']):>8} "
              f"{d['modules']:>8} {d['positive']:>8} {d['clean']:>8}")
        tot["modules"] += d["modules"]; tot["positive"] += d["positive"]
        tot["clean"] += d["clean"]; tot["variants"] += len(d["variants"])
        tot["pos_variants"] += len(d["pos_variants"])
    print(f"{'TOTAL':<8} {tot['variants']:>8} {tot['pos_variants']:>8} "
          f"{tot['modules']:>8} {tot['positive']:>8} {tot['clean']:>8}")
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reconcile paper counts")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--paper-positive", type=int, default=95,
                    help="the Trojan+ count printed in the paper's data table")
    ap.add_argument("--paper-modules", type=int, default=469,
                    help="the total module count printed in the paper's data table")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    proc = cfg["paths"]["processed"]
    if not os.path.exists(proc):
        print(f"[reconcile] processed corpus not found: {proc}")
        print("[reconcile] build it first with 01_prepare_data.py --sources ...")
        return 1

    records = read_jsonl(proc)
    corpus_fam = census(records)
    tot = print_census("CORPUS census (from combined.jsonl)", corpus_fam)

    print(f"\n[paper claims] modules={args.paper_modules}, Trojan+={args.paper_positive}")
    ok_mod = tot["modules"] == args.paper_modules
    ok_pos = tot["positive"] == args.paper_positive
    print(f"[match] modules: {'OK' if ok_mod else 'MISMATCH'} "
          f"({tot['modules']} vs {args.paper_modules})")
    print(f"[match] Trojan+: {'OK' if ok_pos else 'MISMATCH'} "
          f"({tot['positive']} vs {args.paper_positive})")

    # --- cross-check against CV predictions, if present ------------------- #
    runs = cfg["paths"].get("runs", "runs")
    cvp = os.path.join(runs, "cv_predictions.jsonl")
    if not os.path.exists(cvp):
        print(f"\n[reconcile] no CV predictions at {cvp} (run run_experiment.py first)")
        print("[reconcile] corpus census above is still authoritative for the data table.")
        return 0

    preds = read_jsonl(cvp)
    pred_files = {p["file"] for p in preds}
    pred_pos = sum(1 for p in preds if int(p.get("y_true", 0)) == 1)
    tp = sum(1 for p in preds if int(p["y_true"]) == 1 and int(p["y_pred"]) == 1)
    fn = sum(1 for p in preds if int(p["y_true"]) == 1 and int(p["y_pred"]) == 0)
    fp = sum(1 for p in preds if int(p["y_true"]) == 0 and int(p["y_pred"]) == 1)
    tn = sum(1 for p in preds if int(p["y_true"]) == 0 and int(p["y_pred"]) == 0)

    pred_records = [r for r in records if r["file"] in pred_files]
    print_census("EVALUATED census (only designs in cv_predictions.jsonl)",
                 census(pred_records))
    print(f"\n[confusion] tp={tp} fp={fp} fn={fn} tn={tn}  (total {len(preds)})")
    print(f"[confusion] positives evaluated = tp+fn = {tp + fn}")
    print(f"[corpus]    positives in corpus  = {tot['positive']}")

    # The smoking gun: positive designs present in the corpus but NOT evaluated.
    corpus_pos_files = {r["file"] for r in records if int(r.get("label", 0)) == 1}
    missing_pos = sorted(corpus_pos_files - pred_files)
    if missing_pos:
        print(f"\n[GAP] {len(missing_pos)} Trojan-positive design(s) are in the corpus "
              f"but NOT in cv_predictions.jsonl:")
        for f in missing_pos:
            print(f"        - {f}")
        print("[GAP] Likely cause: skipped by a max-length/limit during evaluation, "
              "or an empty test fold. Decide: re-include them (rerun) OR update the "
              "data table + confusion figure to the evaluated counts. Do NOT leave "
              "the two numbers different in the paper.")
    else:
        print("\n[OK] Every corpus positive appears in the predictions; "
              "the 90-vs-95 gap is NOT from dropped designs — check the figure numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
