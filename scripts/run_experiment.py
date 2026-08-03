#!/usr/bin/env python3
"""Real experiment — leave-one-variant-out CV on the Trust-Hub RTL set.

Paper concept
-------------
With only a handful of labeled Trojan variants (e.g. RS232-T2100..T2400), a
full fine-tune would memorize the shared base and overfit. We therefore:

  * FREEZE the Qwen backbone and train only the two lightweight heads
    (detection + per-token localization) on cached backbone features — fast,
    laptop-feasible, and far less prone to overfit on tiny data;
  * evaluate with LEAVE-ONE-VARIANT-OUT cross-validation: for each Trojan
    variant V, train on the *other* variants and score V's unseen Trojan, so the
    model is never tested on a Trojan (or a base+Trojan pair) it trained on;
  * report detection, localization, and the verification rate (VR) aggregated
    over the held-out folds.

This is the honest small-data pilot. Headline numbers need more families
(AES) and are best run with a larger backbone on a GPU — but the protocol and
plumbing are exactly what those runs will use.

Usage
-----
    python scripts/run_experiment.py --config config.yaml --epochs 12
"""
import argparse
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_module(filename: str, name: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def read_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def variant_of(file_path: str) -> str:
    """Top-level benchmark folder, e.g. 'RS232-T2100/src/TjIn/u_rec.v' -> 'RS232-T2100'."""
    return file_path.replace("\\", "/").split("/")[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Leave-one-variant-out TrojanLens experiment")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=12, help="head-training epochs per fold")
    ap.add_argument("--loc-threshold", type=float, default=0.5,
                    help="sigmoid threshold on the localization head for pred_lines")
    ap.add_argument("--processed", default=None,
                    help="override cfg['paths']['processed'] (e.g. an obfuscated "
                         "combined_*.jsonl from 10_obfuscate.py)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip faithfulness verification (fast Det F1 + localization "
                         "only). Use for the obfuscation robustness sweep on CPU.")
    ap.add_argument("--out", default=None, help="override output predictions path")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    import model as M
    explain = load_module("04_explain.py", "explain")
    attributions = load_module("05_attributions.py", "attributions")
    verify = load_module("06_verify.py", "verify")
    metrics = load_module("07_metrics.py", "metrics")

    cfg = load_config(args.config)
    tcfg = cfg.get("train", {})
    lr = float(tcfg.get("learning_rate", 2e-4))
    gamma = float(tcfg.get("focal_gamma", 2.0))
    alpha = float(tcfg.get("focal_alpha", 0.75))
    pos_weight = float(tcfg.get("loc_pos_weight", 8.0))
    top_k = int(cfg.get("verify", {}).get("top_k", 5))
    seed = int(tcfg.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)

    processed_path = args.processed or cfg["paths"]["processed"]
    print(f"[exp] processed = {processed_path}"
          + ("  (verification SKIPPED)" if args.no_verify else ""))
    records = read_jsonl(processed_path)
    recmap = {r["file"]: r for r in records}
    variants = sorted(set(variant_of(r["file"]) for r in records))
    pos_variants = sorted(set(variant_of(r["file"]) for r in records if int(r["label"]) == 1))
    print(f"[exp] {len(records)} designs, {len(variants)} variants, "
          f"{len(pos_variants)} with a Trojan: {pos_variants}")
    if len(pos_variants) < 2:
        print("[exp] Need >=2 Trojan variants for leave-one-out CV. Add more data.")
        return 1

    # ---- build backbone ONCE and freeze it ------------------------------- #
    print("[exp] building backbone (this loads the model once)...")
    net = M.build_model(cfg, toy=False)
    for p in net.encoder.parameters():
        p.requires_grad_(False)
    net.eval()
    H = net.hidden

    # ---- cache per-design backbone features (one forward each) ----------- #
    print("[exp] caching backbone features for all designs...")
    cache = {}
    with torch.no_grad():
        for r in records:
            ids = torch.tensor(r["input_ids"], dtype=torch.long).unsqueeze(0)
            attn = torch.ones_like(ids)
            hidden = net._encode(ids, attn)[0].float()        # [T, H]
            pooled = hidden.mean(dim=0)                        # [H]
            cache[r["file"]] = (hidden, pooled)
    print(f"[exp] cached {len(cache)} designs (hidden size {H}).")

    # ---- leave-one-variant-out CV ---------------------------------------- #
    all_preds = []
    for held in pos_variants:
        train_files = [r["file"] for r in records if variant_of(r["file"]) != held]
        test_files = [r["file"] for r in records if variant_of(r["file"]) == held]

        # fresh heads for this fold
        net.detect_head = nn.Linear(H, 2)
        net.locate_head = nn.Linear(H, 1)
        params = list(net.detect_head.parameters()) + list(net.locate_head.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        net.detect_head.train()
        net.locate_head.train()
        for _ep in range(args.epochs):
            random.shuffle(train_files)
            for f in train_files:
                r = recmap[f]
                hidden, pooled = cache[f]
                det_logits = net.detect_head(pooled.unsqueeze(0))          # [1,2]
                y = torch.tensor([int(r["label"])], dtype=torch.long)
                loss_det = M.focal_loss(det_logits, y, gamma=gamma, alpha=alpha)

                tok_logits = net.locate_head(hidden).squeeze(-1)           # [T]
                tset = set(int(x) for x in r["trojan_lines"])
                tgt = torch.tensor([1.0 if int(tl) in tset else 0.0
                                    for tl in r["token_line"]])
                loss_loc = bce(tok_logits, tgt)

                loss = loss_det + loss_loc
                opt.zero_grad()
                loss.backward()
                opt.step()

        # ---- evaluate held-out variant ---------------------------------- #
        net.detect_head.eval()
        net.locate_head.eval()
        fold_preds = []
        for f in test_files:
            r = recmap[f]
            hidden, pooled = cache[f]
            with torch.no_grad():
                p = torch.softmax(net.detect_head(pooled.unsqueeze(0)), dim=-1)[0, 1].item()
                tok_logits = net.locate_head(hidden).squeeze(-1)
                line_prob = M.aggregate_tokens_to_lines(
                    torch.sigmoid(tok_logits).tolist(), r["token_line"], "mean")
            y_pred = int(p >= 0.5)
            pred_lines = [int(ln) for ln, sc in line_prob.items() if sc >= args.loc_threshold]
            if not pred_lines:  # fall back to top-k by score
                pred_lines = [int(ln) for ln, _ in sorted(
                    line_prob.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]

            verified = False
            if (not args.no_verify) and (y_pred == 1 or int(r["label"]) == 1):
                claim = explain.explain_design(r, cfg, net=net)
                _, topk_attr = attributions.attribute_lines(net, r, cfg)
                verdict = verify.verify_claim(net, r, claim["cited_lines"], topk_attr, cfg)
                verified = bool(verdict["verified"])

            fold_preds.append({
                "file": r["file"], "y_true": int(r["label"]), "y_pred": y_pred,
                "n_lines": int(r["n_lines"]), "gt_lines": list(r["trojan_lines"]),
                "pred_lines": pred_lines, "verified": verified,
                "trojan_prob": round(float(p), 4),
            })

        fd = metrics.detection_metrics(fold_preds)
        print(f"[fold {held}] test={len(fold_preds)} "
              f"det_F1={fd['f1']:.3f} (tp{fd['tp']} fp{fd['fp']} fn{fd['fn']} tn{fd['tn']})")
        all_preds.extend(fold_preds)

    # ---- aggregate ------------------------------------------------------- #
    runs = cfg["paths"]["runs"]
    os.makedirs(runs, exist_ok=True)
    if args.out:
        out = args.out
    elif args.processed:
        stem = os.path.splitext(os.path.basename(args.processed))[0]
        out = os.path.join(runs, f"cv_predictions_{stem}.jsonl")
    else:
        out = os.path.join(runs, "cv_predictions.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for pr in all_preds:
            fh.write(json.dumps(pr) + "\n")

    print("\n=== Leave-one-variant-out aggregate ===")
    m = metrics.compute_all(all_preds)
    metrics.print_table(m)
    print(f"\n[exp] wrote per-design CV predictions -> {out}")
    print("[exp] NOTE: small-data pilot (RS232 only). Add AES + a GPU backbone "
          "for headline numbers; the protocol is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
