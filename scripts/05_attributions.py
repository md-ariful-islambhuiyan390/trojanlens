#!/usr/bin/env python3
"""Step 7 — per-line integrated-gradient attributions for the detection call.

Paper concept
-------------
To *verify* an explanation we first need an independent, model-internal signal
of which lines actually drove the "Trojan" decision. We use **integrated
gradients** on the embedding layer (captum ``LayerIntegratedGradients``),
targeting the Trojan detection logit, then aggregate token attributions up to
source lines via the line<->token map. The result — per-line scores and the
top-k lines — feeds the "attribution grounding" check in Step 8.

Fallback: if captum is not installed we compute a simple ``gradient x input``
saliency on the embedding output instead, which yields comparable per-token
signal for the toy demo.
"""
import argparse
import json
import os
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
# Token attributions
# --------------------------------------------------------------------------- #
def _ig_captum(net, ids2d, attn, n_steps=16, internal_batch_size=1):
    """LayerIntegratedGradients over the embedding layer. Returns [T] tensor.

    ``internal_batch_size`` caps how many integration steps are materialized at
    once. On a laptop keep it small (1) so memory stays flat regardless of
    sequence length; raise it on a GPU for speed.
    """
    import torch
    from captum.attr import LayerIntegratedGradients

    def forward_func(input_ids, attention_mask):
        return net(input_ids, attention_mask)["det_logits"]

    lig = LayerIntegratedGradients(forward_func, net.embedding_layer())
    baseline = torch.zeros_like(ids2d)   # pad-token baseline
    atts = lig.attribute(
        inputs=ids2d,
        baselines=baseline,
        additional_forward_args=(attn,),
        target=1,                        # the "Trojan present" class
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    # atts: [B, T, H] -> sum over hidden dim -> per-token attribution.
    return atts.sum(dim=-1)[0].detach().cpu()


def _grad_x_input(net, ids2d, attn):
    """Fallback saliency: (grad of trojan logit w.r.t. embeddings) * embeddings.
    Returns a [T] tensor."""
    import torch

    emb_layer = net.embedding_layer()
    captured = {}

    def hook(_module, _inp, output):
        output.requires_grad_(True)  # frozen backbone: enable grad so retain_grad works
        output.retain_grad()
        captured["emb"] = output
        return output

    handle = emb_layer.register_forward_hook(hook)
    try:
        net.zero_grad(set_to_none=True)
        out = net(ids2d, attn)
        logit = out["det_logits"][0, 1]      # Trojan-class logit
        logit.backward()
        emb = captured["emb"]                 # [B, T, H]
        sal = (emb * emb.grad).sum(dim=-1)[0]  # [T]
        return sal.detach().cpu()
    finally:
        handle.remove()


def attribute_lines(net, rec, cfg):
    """Return (line_scores: dict[int,float], topk: list[int])."""
    import torch

    ids = torch.tensor(rec["input_ids"], dtype=torch.long)
    ids2d = ids.unsqueeze(0)
    attn = torch.ones_like(ids2d)

    # Fast single-pass attribution by default (works with a frozen backbone and is
    # ~n_steps faster than IG). Set cfg['verify']['attribution']='ig' for full
    # integrated gradients (recommended on GPU where the extra passes are cheap).
    use_ig = str(cfg.get("verify", {}).get("attribution", "gradxinput")).lower() == "ig"
    try:
        if use_ig:
            tok_attr = _ig_captum(net, ids2d, attn)
            method = "integrated_gradients(captum)"
        else:
            tok_attr = _grad_x_input(net, ids2d, attn)
            method = "grad_x_input(fast)"
    except Exception as exc:  # noqa: BLE001
        print(f"[attr] primary attribution failed ({exc}); using grad*input")
        tok_attr = _grad_x_input(net, ids2d, attn)
        method = "grad_x_input(fallback)"

    import model as M
    # Use absolute attribution magnitude aggregated (sum) per line.
    tok_abs = [abs(float(x)) for x in tok_attr.tolist()]
    line_scores = M.aggregate_tokens_to_lines(tok_abs, rec["token_line"], "sum")

    top_k = int(cfg.get("verify", {}).get("top_k", 5))
    ranked = sorted(line_scores.items(), key=lambda kv: kv[1], reverse=True)
    topk = [int(ln) for ln, _ in ranked[:top_k]]
    print(f"[attr] {rec['file']}: method={method} top{top_k}={topk}")
    return line_scores, topk


def main(argv=None):
    ap = argparse.ArgumentParser(description="Integrated-gradient attributions")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the toy model (no weight download)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    recs = read_jsonl(cfg["paths"]["processed"])

    import model as M
    net = M.build_model(cfg, toy=args.dry_run)
    net.eval()

    results = []
    for rec in recs:
        if int(rec.get("label", 0)) != 1:
            continue
        line_scores, topk = attribute_lines(net, rec, cfg)
        results.append({
            "file": rec["file"],
            "top_k_lines": topk,
            "line_scores": {str(k): v for k, v in line_scores.items()},
        })

    out_path = os.path.join(cfg["paths"]["runs"], "attributions.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"[attr] wrote {len(results)} attribution record(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
