#!/usr/bin/env python3
"""Step 6 — generate a natural-language explanation that CITES suspect lines.

Paper concept
-------------
For any design the detector flags positive, TrojanLens must emit a *claim*: a
human-readable explanation together with the specific source lines it blames.
That claim is exactly what the verification layer (Steps 7-8) then checks for
faithfulness. The explanation must be a structured object::

    {"explanation": str, "cited_lines": [int, ...]}

Two ways to produce ``cited_lines`` are provided:
  * ``model``     — aggregate the localization head's per-token logits up to
    lines and cite the top ones (the intended, learned path); and
  * ``heuristic`` — structural RTL red-flags (rare magic-constant comparisons =
    triggers; inverted output assignments = payloads). Used as an offline
    fallback and as a sanity check.

TODO(you): swap/augment the template with a real prompt to the base Qwen model
(or the GGUF 7B baseline) to produce richer prose; keep the structured
``cited_lines`` contract so verification still applies.
"""
import argparse
import json
import os
import re
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


# --------------------------------------------------------------------------- #
# Structural heuristics (offline fallback / sanity check)
# --------------------------------------------------------------------------- #
_MAGIC_CMP = re.compile(r"==\s*\d+'\s*[hHbB][0-9a-fA-F_]+")   # e.g. == 24'hADBEEF
_INVERT_ASSIGN = re.compile(r"<=\s*~")                        # e.g. tx <= ~data


def explain_heuristic(rec: dict):
    """Cite lines matching known Trojan signatures. 1-indexed line numbers.

    We deliberately use only high-precision structural signatures here — a rare
    magic-constant comparison (a classic trigger) and an inverted assignment to
    an output (a corrupting payload). Broader rules (e.g. any flag latched high)
    match too much legitimate RTL; the *learned* localization head (see
    ``explain_from_model``) is the intended, higher-recall path.
    """
    cited = []
    reasons = []
    lines = rec["lines"]
    for i, line in enumerate(lines, start=1):
        if _MAGIC_CMP.search(line):
            cited.append(i)
            reasons.append(f"line {i}: comparison against a rare magic constant "
                           f"(classic trigger)")
        elif _INVERT_ASSIGN.search(line):
            cited.append(i)
            reasons.append(f"line {i}: an output is driven by an inverted signal "
                           f"(payload corrupting behavior)")
    return sorted(set(cited)), reasons


# --------------------------------------------------------------------------- #
# Model-driven citation (intended path)
# --------------------------------------------------------------------------- #
def explain_from_model(net, rec: dict, top_k: int):
    """Cite the top-k lines by the localization head's aggregated logits."""
    import torch
    import model as M

    ids = torch.tensor(rec["input_ids"], dtype=torch.long)
    attn = torch.ones_like(ids)
    with torch.no_grad():
        out = net(ids, attn)
    tok_logits = out["tok_logits"][0].detach().cpu().tolist()
    line_scores = M.aggregate_tokens_to_lines(tok_logits, rec["token_line"], "mean")
    ranked = sorted(line_scores.items(), key=lambda kv: kv[1], reverse=True)
    cited = sorted(int(ln) for ln, _ in ranked[:top_k])
    reasons = [f"line {ln}: high localization-head activation ({sc:.3f})"
               for ln, sc in ranked[:top_k]]
    return cited, reasons


def build_explanation_text(rec: dict, cited, reasons) -> str:
    head = (f"Design '{rec['file']}' is flagged as containing a hardware Trojan. "
            f"The suspected malicious logic is localized to line(s) "
            f"{cited}. ")
    body = " ".join(reasons) if reasons else "No specific signature lines found."
    tail = (" The trigger is a rarely-satisfied condition that arms a payload "
            "which corrupts the module's intended output.")
    return head + body + tail


def explain_design(rec: dict, cfg: dict, net=None):
    """Return the structured claim {explanation, cited_lines}."""
    top_k = int(cfg.get("verify", {}).get("top_k", 5))
    if net is not None:
        cited, reasons = explain_from_model(net, rec, top_k)
    else:
        cited, reasons = explain_heuristic(rec)
    return {
        "file": rec["file"],
        "cited_lines": cited,
        "explanation": build_explanation_text(rec, cited, reasons),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Explain flagged designs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--heuristic-only", action="store_true",
                    help="use structural heuristics instead of the model head")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the toy model for the model-driven path")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    recs = read_jsonl(cfg["paths"]["processed"])

    net = None
    if not args.heuristic_only:
        try:
            import model as M
            net = M.build_model(cfg, toy=args.dry_run)
            net.eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[explain] model unavailable ({exc}); using heuristics")
            net = None

    claims = []
    for rec in recs:
        # Only explain designs predicted/known positive. Here we use the label
        # as the positive gate for the toy; in production use the detector.
        if int(rec.get("label", 0)) != 1:
            continue
        claim = explain_design(rec, cfg, net=net)
        claims.append(claim)
        print(json.dumps(claim, indent=2))

    out_path = os.path.join(cfg["paths"]["runs"], "explanations.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for c in claims:
            fh.write(json.dumps(c) + "\n")
    print(f"[explain] wrote {len(claims)} claim(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
