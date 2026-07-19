#!/usr/bin/env python3
"""One-command toy end-to-end run of the whole TrojanLens flow.

Pipeline: prepare -> (toy/loaded) model -> explain -> attributions -> verify
-> metrics. This ties every mechanism together on the bundled toy dataset so
you can see the paper's loop working before wiring in Trust-Hub.

Modes
-----
  --dry-run : build the *toy* encoder (no Qwen download) so the logic is fully
              demonstrable offline. Numbers from a random toy model are NOT
              meaningful results — they only prove the plumbing is coherent.
  (default) : use the real Qwen2.5-Coder-1.5B backbone (downloads weights).

The numbered step scripts (01/04/05/06/07) are loaded by file path because
their names start with digits.
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_module(filename: str, name: str):
    """Import a sibling script (even if its filename starts with a digit)."""
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(description="Toy end-to-end TrojanLens run")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="toy model, no weight download (offline demo)")
    args = ap.parse_args(argv)

    prepare = load_module("01_prepare_data.py", "prepare_data")
    explain = load_module("04_explain.py", "explain")
    attributions = load_module("05_attributions.py", "attributions")
    verify = load_module("06_verify.py", "verify")
    metrics = load_module("07_metrics.py", "metrics")
    import model as M

    cfg = prepare.load_config(args.config)
    runs = cfg["paths"]["runs"]
    os.makedirs(runs, exist_ok=True)

    # ---- Step 4: prepare -------------------------------------------------- #
    print("\n=== [1/5] prepare data ===")
    records = prepare.build_dataset(cfg)
    prepare.write_jsonl(records, cfg["paths"]["processed"])

    # ---- model ------------------------------------------------------------ #
    print("\n=== build model ===")
    net = M.build_model(cfg, toy=args.dry_run)
    net.eval()
    if args.dry_run:
        print("[pipeline] using TOY model (random weights) — logic demo only.")

    # ---- per-design explain + attribute + verify ------------------------- #
    predictions = []
    for rec in records:
        import torch
        ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        attn = torch.ones_like(ids)
        p = net.trojan_prob(ids, attn)
        y_pred = int(p >= 0.5)

        pred_lines, verified = [], False
        if y_pred == 1 or int(rec["label"]) == 1:
            # Produce a claim, attribute, and verify it.
            claim = explain.explain_design(rec, cfg, net=net)
            _, topk = attributions.attribute_lines(net, rec, cfg)
            verdict = verify.verify_claim(net, rec, claim["cited_lines"], topk, cfg)
            pred_lines = claim["cited_lines"]
            verified = bool(verdict["verified"])

        predictions.append({
            "file": rec["file"],
            "y_true": int(rec["label"]),
            "y_pred": y_pred,
            "n_lines": int(rec["n_lines"]),
            "gt_lines": list(rec["trojan_lines"]),
            "pred_lines": pred_lines,
            "verified": verified,
            "trojan_prob": round(float(p), 4),
        })

    pred_path = cfg["paths"]["predictions"]
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as fh:
        for pr in predictions:
            fh.write(json.dumps(pr) + "\n")
    print(f"\n[pipeline] wrote predictions -> {pred_path}")

    # ---- Step 9: metrics -------------------------------------------------- #
    print("\n=== [5/5] metrics ===")
    m = metrics.compute_all(predictions)
    metrics.print_table(m)

    if args.dry_run:
        print("\nReminder: --dry-run uses a random toy model; the numbers above "
              "demonstrate the pipeline wiring, not detector quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
