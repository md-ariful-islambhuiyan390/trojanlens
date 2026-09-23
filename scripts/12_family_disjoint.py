#!/usr/bin/env python3
"""Family-disjoint generalization (revision, Reviewer M1/M3/R2-1).

Trains on ALL records of one family and evaluates on the ENTIRELY held-out other
family -- direct evidence for generalization to unseen IP families (unlike the
leave-one-variant-out protocol, where other variants of the same family remain in
training).

Usage
-----
    python scripts/12_family_disjoint.py --config config.yaml \
        --train-family AES --test-family RS232 --out runs/fd_aes2rs232.jsonl
    python scripts/12_family_disjoint.py --config config.yaml \
        --train-family RS232 --test-family AES --out runs/fd_rs2322aes.jsonl

Reuses the same frozen-backbone + two-head recipe as run_experiment.py so the
numbers are directly comparable to the paper's LOO results.
"""
import argparse
import importlib.util
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_module(filename, name):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    """'RS232-T2100/src/.../u_rec.v' -> 'RS232-T2100'."""
    return file_path.replace("\\", "/").split("/")[0]


def family_of(file_path):
    """'RS232-T2100' -> 'RS232'; 'AES_T1000' -> 'AES'. Split on first '-' or '_'."""
    return re.split(r"[-_]", variant_of(file_path))[0].upper()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Family-disjoint TrojanLens experiment")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--train-family", required=True, help="e.g. AES")
    ap.add_argument("--test-family", required=True, help="e.g. RS232")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--loc-threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=None, help="override focal-loss alpha")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--processed", default=None)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--out", default=None)
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
    alpha = args.alpha if args.alpha is not None else float(tcfg.get("focal_alpha", 0.75))
    pos_weight = float(tcfg.get("loc_pos_weight", 8.0))
    top_k = int(cfg.get("verify", {}).get("top_k", 5))
    seed = args.seed if args.seed is not None else int(tcfg.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)

    train_fam = args.train_family.upper()
    test_fam = args.test_family.upper()
    processed_path = args.processed or cfg["paths"]["processed"]
    records = read_jsonl(processed_path)
    recmap = {r["file"]: r for r in records}

    train_files = [r["file"] for r in records if family_of(r["file"]) == train_fam]
    test_files = [r["file"] for r in records if family_of(r["file"]) == test_fam]
    fams = sorted(set(family_of(r["file"]) for r in records))
    print(f"[fd] families present: {fams}")
    print(f"[fd] train={train_fam} ({len(train_files)} recs)  "
          f"test={test_fam} ({len(test_files)} recs)  seed={seed} alpha={alpha}")
    if not train_files or not test_files:
        print("[fd] ERROR: empty train or test set -- check family names against the list above.")
        return 1

    print("[fd] building + freezing backbone...")
    net = M.build_model(cfg, toy=False)
    for p in net.encoder.parameters():
        p.requires_grad_(False)
    net.eval()
    H = net.hidden

    cache = {}
    with torch.no_grad():
        for r in records:
            ids = torch.tensor(r["input_ids"], dtype=torch.long).unsqueeze(0)
            attn = torch.ones_like(ids)
            hidden = net._encode(ids, attn)[0].float()
            cache[r["file"]] = (hidden, hidden.mean(dim=0))

    net.detect_head = nn.Linear(H, 2)
    net.locate_head = nn.Linear(H, 1)
    params = list(net.detect_head.parameters()) + list(net.locate_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    net.detect_head.train(); net.locate_head.train()
    for _ep in range(args.epochs):
        random.shuffle(train_files)
        for f in train_files:
            r = recmap[f]
            hidden, pooled = cache[f]
            det_logits = net.detect_head(pooled.unsqueeze(0))
            y = torch.tensor([int(r["label"])], dtype=torch.long)
            loss_det = M.focal_loss(det_logits, y, gamma=gamma, alpha=alpha)
            tok_logits = net.locate_head(hidden).squeeze(-1)
            tset = set(int(x) for x in r["trojan_lines"])
            tgt = torch.tensor([1.0 if int(tl) in tset else 0.0 for tl in r["token_line"]])
            loss = loss_det + bce(tok_logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    net.detect_head.eval(); net.locate_head.eval()
    preds = []
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
        if not pred_lines:
            pred_lines = [int(ln) for ln, _ in sorted(
                line_prob.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        verified = False
        if (not args.no_verify) and (y_pred == 1 or int(r["label"]) == 1):
            claim = explain.explain_design(r, cfg, net=net)
            _, topk_attr = attributions.attribute_lines(net, r, cfg)
            verdict = verify.verify_claim(net, r, claim["cited_lines"], topk_attr, cfg)
            verified = bool(verdict["verified"])
        preds.append({
            "file": r["file"], "y_true": int(r["label"]), "y_pred": y_pred,
            "n_lines": int(r["n_lines"]), "gt_lines": list(r["trojan_lines"]),
            "pred_lines": pred_lines, "verified": verified,
            "trojan_prob": round(float(p), 4),
        })

    runs = cfg["paths"]["runs"]; os.makedirs(runs, exist_ok=True)
    out = args.out or os.path.join(runs, f"fd_{train_fam}2{test_fam}.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for pr in preds:
            fh.write(json.dumps(pr) + "\n")

    print(f"\n=== Family-disjoint: train {train_fam} -> test {test_fam} ===")
    m = metrics.compute_all(preds)
    metrics.print_table(m)
    print(f"[fd] wrote predictions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
