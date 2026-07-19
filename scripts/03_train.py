#!/usr/bin/env python3
"""Step 5 — train the detection + localization heads (LoRA backbone).

Paper concept
-------------
We jointly train:
  * detection  (module-level Trojan present?) with **focal loss** (the Trojan
    class is rare -> class imbalance), and
  * localization (which lines are the Trojan?) with a **pos-weighted
    binary cross-entropy** over per-token logits (Trojan tokens are sparse).

Only LoRA adapter params + the two lightweight heads are trained; the Qwen
backbone stays frozen. On an M1 Pro this is slow-but-feasible for small runs.

Modes
-----
  --dry-run : build the *toy* encoder and train on random tensors. No weights
              are downloaded; this only proves the training logic wires up.
  (default) : load the real Qwen backbone + LoRA and train on the processed
              toy dataset.
"""
import argparse
import json
import os
import sys

# Make sibling ``model.py`` importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        raise SystemExit("pyyaml not installed: pip install pyyaml")


def read_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def pick_device(requested: str):
    import torch
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested not in ("cpu",):
        print(f"[train] requested device '{requested}' unavailable; using cpu")
    return "cpu"


def build_targets(rec, n_tokens):
    """Per-token localization target: 1 if the token's line is a Trojan line."""
    trojan = set(int(x) for x in rec.get("trojan_lines", []))
    token_line = rec["token_line"]
    return [1.0 if int(token_line[i]) in trojan else 0.0 for i in range(n_tokens)]


def train(args):
    import torch
    from torch.optim import AdamW
    import torch.nn.functional as F
    import model as M

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    epochs = args.epochs if args.epochs is not None else int(tcfg["epochs"])
    torch.manual_seed(int(tcfg.get("seed", 1234)))

    device = pick_device(cfg["model"].get("device", "cpu")) if not args.dry_run else "cpu"
    net = M.build_model(cfg, toy=args.dry_run)
    net.to(device)
    net.train()

    # Only train parameters that require grad (LoRA adapters + heads).
    params = [p for p in net.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"[train] device={device} trainable_params={n_train:,} "
          f"mode={'dry-run(toy)' if args.dry_run else 'real'}")
    opt = AdamW(params, lr=float(tcfg["learning_rate"]))

    pos_weight = torch.tensor([float(tcfg.get("loc_pos_weight", 8.0))], device=device)
    gamma = float(tcfg.get("focal_gamma", 2.0))
    alpha = float(tcfg.get("focal_alpha", 0.75))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    # Build the batch list.
    if args.dry_run:
        # Random tiny "designs": 2 examples, one clean, one trojan.
        batches = []
        for label in (0, 1):
            T = 16
            ids = torch.randint(0, 500, (T,))
            token_line = [1 + i // 4 for i in range(T)]     # 4 tokens per line
            trojan_lines = [2, 3] if label == 1 else []
            batches.append({
                "input_ids": ids.tolist(),
                "token_line": token_line,
                "trojan_lines": trojan_lines,
                "label": label,
            })
    else:
        batches = read_jsonl(cfg["paths"]["processed"])

    for epoch in range(epochs):
        total = 0.0
        for rec in batches:
            ids = torch.tensor(rec["input_ids"], dtype=torch.long, device=device)
            attn = torch.ones_like(ids)
            out = net(ids, attn)
            det_logits = out["det_logits"]                  # [1, 2]
            tok_logits = out["tok_logits"]                  # [1, T]

            det_target = torch.tensor([int(rec["label"])], device=device)
            det_loss = M.focal_loss(det_logits, det_target, gamma=gamma, alpha=alpha)

            loc_target = torch.tensor(
                [build_targets(rec, tok_logits.shape[-1])], device=device
            )
            loc_loss = F.binary_cross_entropy_with_logits(
                tok_logits, loc_target, pos_weight=pos_weight
            )

            loss = det_loss + loc_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            total += float(loss.detach().cpu())
        print(f"[train] epoch {epoch + 1}/{epochs}  loss={total / len(batches):.4f}")

    save_dir = cfg["paths"]["runs"]
    os.makedirs(save_dir, exist_ok=True)
    if args.dry_run:
        torch.save(net.state_dict(), os.path.join(save_dir, "toy_model.pt"))
        print(f"[train] dry-run complete; toy weights -> {save_dir}/toy_model.pt")
    else:
        # Save LoRA adapter (peft) + the two heads.
        try:
            net.encoder.save_pretrained(os.path.join(save_dir, "adapter"))
        except Exception as exc:  # noqa: BLE001
            print(f"[train] warning: could not save adapter ({exc})")
        torch.save(
            {"detect_head": net.detect_head.state_dict(),
             "locate_head": net.locate_head.state_dict()},
            os.path.join(save_dir, "heads.pt"),
        )
        print(f"[train] saved adapter + heads -> {save_dir}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train TrojanLens heads (LoRA)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="use toy model + random tensors (no weight download)")
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
