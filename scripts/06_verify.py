#!/usr/bin/env python3
"""Step 8 — faithfulness verification of an explanation (the novel layer).

Paper concept
-------------
An explanation is only trustworthy if it is *faithful* to the model. Given the
cited lines (Step 6), the integrated-gradient attributions (Step 7), and the
detector, we run three checks:

  (a) Attribution grounding
        AG = |topk(attr) ∩ cited| / |cited|            must be >= tau_g
      The cited lines should coincide with the lines IG says mattered.

  (b) Comprehensiveness (counterfactual removal)
        neutralize the cited lines -> detection prob must DROP by >= tau_c
        AND the prediction must FLIP to clean (0).
      If deleting the cited logic makes the design look clean, the citation
      really was carrying the decision.

  (c) Control / specificity
        neutralize a random equal-size NON-cited span -> prediction must NOT
        flip (control_held). This rules out "masking anything flips it".

Verdict::

    {"verified", "AG", "delta_c", "delta_s", "flip", "control_held"}

``verified`` is True iff all three checks pass.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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


def _prob_with_lines_masked(net, rec, lines_to_mask, mask_id=0):
    """Detection Trojan-probability after neutralizing the given source lines."""
    import torch
    import model as M
    masked_ids = M.neutralize_lines(
        rec["input_ids"], rec["token_line"], lines_to_mask, mask_id=mask_id
    )
    ids = torch.tensor(masked_ids, dtype=torch.long)
    attn = torch.ones_like(ids)
    return net.trojan_prob(ids, attn)


def _pick_control_lines(rec, cited_lines, seed=0):
    """Random equal-size set of non-cited lines that actually contain tokens."""
    present = [int(ln) for ln, span in rec["line_spans"].items()
               if span[1] > span[0]]
    candidates = [ln for ln in present if ln not in set(cited_lines)]
    rng = random.Random(seed)
    k = min(len(cited_lines), len(candidates))
    if k == 0:
        return []
    return rng.sample(candidates, k)


def verify_claim(net, rec, cited_lines, topk_attr_lines, cfg):
    """Run all three checks and return the verdict dict."""
    import torch
    vcfg = cfg.get("verify", {})
    tau_g = float(vcfg.get("tau_g", 0.5))
    tau_c = float(vcfg.get("tau_c", 0.2))
    seed = int(cfg.get("train", {}).get("seed", 1234))

    cited = [int(x) for x in cited_lines]

    # Baseline (unmasked) prediction.
    ids = torch.tensor(rec["input_ids"], dtype=torch.long)
    attn = torch.ones_like(ids)
    p_orig = net.trojan_prob(ids, attn)

    # (a) attribution grounding
    inter = set(cited) & set(int(x) for x in topk_attr_lines)
    AG = (len(inter) / len(cited)) if cited else 0.0

    # (b) comprehensiveness: mask the cited lines
    p_cited = _prob_with_lines_masked(net, rec, cited)
    delta_c = p_orig - p_cited
    flip = p_cited < 0.5 and p_orig >= 0.5

    # (c) control/specificity: mask a random equal-size non-cited span
    control_lines = _pick_control_lines(rec, cited, seed=seed)
    p_control = _prob_with_lines_masked(net, rec, control_lines)
    delta_s = p_orig - p_control
    control_flip = p_control < 0.5 and p_orig >= 0.5
    control_held = not control_flip

    verified = bool(AG >= tau_g and (delta_c >= tau_c and flip) and control_held)

    return {
        "file": rec["file"],
        "verified": verified,
        "AG": round(AG, 4),
        "delta_c": round(float(delta_c), 4),
        "delta_s": round(float(delta_s), 4),
        "flip": bool(flip),
        "control_held": bool(control_held),
        "cited_lines": cited,
        "topk_attr_lines": [int(x) for x in topk_attr_lines],
        "control_lines": control_lines,
        "p_orig": round(float(p_orig), 4),
        "p_cited_masked": round(float(p_cited), 4),
        "p_control_masked": round(float(p_control), 4),
        "thresholds": {"tau_g": tau_g, "tau_c": tau_c},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify explanation faithfulness")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    recs = {r["file"]: r for r in read_jsonl(cfg["paths"]["processed"])}

    runs = cfg["paths"]["runs"]
    claims = read_jsonl(os.path.join(runs, "explanations.jsonl"))
    attrs = {a["file"]: a for a in read_jsonl(os.path.join(runs, "attributions.jsonl"))}

    import model as M
    net = M.build_model(cfg, toy=args.dry_run)
    net.eval()

    verdicts = []
    for claim in claims:
        f = claim["file"]
        rec = recs[f]
        topk = attrs.get(f, {}).get("top_k_lines", [])
        v = verify_claim(net, rec, claim["cited_lines"], topk, cfg)
        verdicts.append(v)
        print(json.dumps(v, indent=2))

    out_path = os.path.join(runs, "verdicts.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v) + "\n")
    print(f"[verify] wrote {len(verdicts)} verdict(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
