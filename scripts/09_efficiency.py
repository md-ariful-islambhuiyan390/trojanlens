#!/usr/bin/env python3
"""Phase-1 efficiency measurement — fills the 'Efficiency' row of Table 8
(per-design latency + peak memory) with real numbers.

Reviewer 1, H4 / point 1: the paper promises a commodity per-design budget but
leaves latency/memory as '---'. This measures them on YOUR machine.

We report, over the corpus:
  * detection-only latency  (one forward pass, the common case for clean IP);
  * full verified-finding latency (forward + explain + attributions + the
    3-check verification), i.e. the extra cost incurred only on positives;
  * peak resident memory (RSS) and, if on GPU/MPS, peak device memory.

Usage
-----
    # real model on your M1 (downloads Qwen 0.5B the first time):
    python scripts/09_efficiency.py --config config.yaml --limit 60
    # smoke test with the offline toy encoder (no download):
    python scripts/09_efficiency.py --config config.yaml --dry-run --limit 20

Paste the printed 'PAPER-READY' block into Table 8.
"""
import argparse
import importlib.util
import json
import os
import resource
import statistics
import sys
import time

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


def peak_rss_gb():
    """Peak resident set size in GB. ru_maxrss is bytes on macOS, KiB on Linux."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor = 1 if sys.platform == "darwin" else 1024  # macOS bytes vs Linux KiB
    return ru * factor / (1024 ** 3)


def device_peak_gb(torch):
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3), "cuda"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # driver_allocated_memory added in torch 2.3; fall back gracefully.
            fn = getattr(torch.mps, "driver_allocated_memory", None) \
                or getattr(torch.mps, "current_allocated_memory", None)
            if fn:
                return fn() / (1024 ** 3), "mps"
    except Exception:
        pass
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description="TrojanLens efficiency measurement")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="use the toy encoder")
    ap.add_argument("--limit", type=int, default=60,
                    help="cap designs timed (keeps it quick; 0 = all)")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args(argv)

    import torch
    import model as M
    explain = load_module("04_explain.py", "explain")
    attributions = load_module("05_attributions.py", "attributions")
    verify = load_module("06_verify.py", "verify")

    cfg = load_config(args.config)
    records = read_jsonl(cfg["paths"]["processed"])
    if args.limit and len(records) > args.limit:
        # keep a mix of positives and negatives
        pos = [r for r in records if int(r.get("label", 0)) == 1]
        neg = [r for r in records if int(r.get("label", 0)) == 0]
        half = args.limit // 2
        records = (pos[:half] + neg[:args.limit - min(half, len(pos))])
    print(f"[eff] timing {len(records)} designs "
          f"({sum(int(r.get('label',0))==1 for r in records)} positive) "
          f"| dry_run={args.dry_run}")

    net = M.build_model(cfg, toy=args.dry_run)
    net.eval()

    def detect_once(rec):
        ids = torch.tensor(rec["input_ids"], dtype=torch.long)
        attn = torch.ones_like(ids)
        with torch.no_grad():
            _ = net.trojan_prob(ids, attn)

    def verify_once(rec):
        claim = explain.explain_design(rec, cfg, net=net)
        _, topk = attributions.attribute_lines(net, rec, cfg)
        _ = verify.verify_claim(net, rec, claim["cited_lines"], topk, cfg)

    # warmup (excluded from timing)
    for r in records[:args.warmup]:
        detect_once(r)

    det_times, ver_times = [], []
    for r in records:
        t0 = time.perf_counter(); detect_once(r); det_times.append(time.perf_counter() - t0)
        if int(r.get("label", 0)) == 1:
            t0 = time.perf_counter(); verify_once(r); ver_times.append(time.perf_counter() - t0)

    rss = peak_rss_gb()
    dev_gb, dev_name = device_peak_gb(torch)

    def summ(xs):
        if not xs:
            return None
        return (statistics.median(xs), min(xs), max(xs))

    d = summ(det_times); v = summ(ver_times)
    print("\n=== raw timing (seconds/design) ===")
    if d:
        print(f"detection : median {d[0]*1000:.1f} ms  (min {d[1]*1000:.1f}, max {d[2]*1000:.1f})")
    if v:
        print(f"verify    : median {v[0]:.3f} s   (min {v[1]:.3f}, max {v[2]:.3f})")
    else:
        print("verify    : no positive designs in the timed subset")
    print(f"peak RSS  : {rss:.2f} GB"
          + (f" | peak {dev_name} device mem {dev_gb:.2f} GB" if dev_gb else ""))

    # paper-ready block for Table 8
    print("\n=== PAPER-READY (Table 8 efficiency row) ===")
    det_lat = d[0] if d else 0.0
    ver_lat = v[0] if v else None
    # RSS is the reliable figure; MPS 'device' memory is unified and often
    # reports ~0, so take the larger of the two.
    mem = max(rss, dev_gb or 0.0)
    dev_label = cfg.get("model", {}).get("device", "cpu")
    ver_str = f"; full verified finding (flagged designs) ~{ver_lat:.0f}\\,s" if ver_lat else ""
    print(f"Efficiency: detection ~{det_lat:.1f}\\,s/design{ver_str}; "
          f"peak memory ~{mem:.1f}\\,GB ({dev_label}).")
    print("NOTE: numbers are from THIS run on THIS machine; re-run on the M1 Pro "
          "you cite in the paper before quoting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
