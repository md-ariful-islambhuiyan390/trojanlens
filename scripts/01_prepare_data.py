#!/usr/bin/env python3
"""Step 4 — preprocess RTL into a line-aware tokenized dataset.

Paper concept
-------------
TrojanLens must *localize* Trojans to specific source lines and later
*neutralize* those lines for counterfactual verification. Both operations
require a faithful **line <-> token map**: for every token we must know which
1-indexed source line it came from, and for every source line we must know its
contiguous token span.

This script:
  1. reads designs + ``labels.jsonl``,
  2. tokenizes each design **one source line at a time** (so spans never cross
     a line boundary),
  3. records ``token_line`` (per-token line number) and ``line_spans``
     (line number -> [start, end) token indices),
  4. writes one JSON record per design to the processed dataset.

It also ingests real **Trust-Hub** RTL benchmarks: ``parse_trusthub()`` pairs
Trojan-free and Trojan-in Verilog, derives Trojan lines by diff, and writes a
``labels.jsonl`` in the same format the toy path uses (``--trusthub`` CLI).
"""
import argparse
import difflib
import json
import os
import re
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Config helper
# --------------------------------------------------------------------------- #
def load_config(path: str) -> dict:
    """Load YAML config; fall back to a tiny hand-parsed subset if pyyaml
    is missing so the script still runs offline."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        return {
            "model": {"name": "Qwen/Qwen2.5-Coder-0.5B", "max_seq_len": 1024},
            "paths": {
                "toy_dir": "data/toy",
                "labels": "data/toy/labels.jsonl",
                "processed": "data/processed/toy.jsonl",
            },
        }


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
class FallbackTokenizer:
    """Deterministic tokenizer used when transformers is unavailable."""

    _PATTERN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s]")

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}

    @property
    def pad_id(self) -> int:
        return 0

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]

    def encode_line(self, line: str):
        toks = self._PATTERN.findall(line)
        ids = [self._id(t) for t in toks]
        return ids, toks


def build_tokenizer(model_name: str):
    """Return (tokenizer_obj, kind) where kind is 'hf' or 'fallback'."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        return tok, "hf"
    except Exception:
        return FallbackTokenizer(), "fallback"


# --------------------------------------------------------------------------- #
# Core line<->token mapping
# --------------------------------------------------------------------------- #
def tokenize_with_line_map(source: str, tokenizer, kind: str, max_seq_len: int):
    """Tokenize ``source`` line by line and build the line<->token map."""
    lines = source.splitlines()
    input_ids: List[int] = []
    token_strings: List[str] = []
    token_line: List[int] = []
    line_spans: Dict[str, List[int]] = {}

    for idx, line in enumerate(lines, start=1):
        start = len(input_ids)
        if kind == "hf":
            ids = tokenizer.encode(line, add_special_tokens=False)
            surfaces = tokenizer.convert_ids_to_tokens(ids)
        else:
            ids, surfaces = tokenizer.encode_line(line)

        if len(input_ids) + len(ids) > max_seq_len:
            ids = ids[: max(0, max_seq_len - len(input_ids))]
            surfaces = surfaces[: len(ids)]

        input_ids.extend(ids)
        token_strings.extend(surfaces)
        token_line.extend([idx] * len(ids))
        end = len(input_ids)
        line_spans[str(idx)] = [start, end]

        if len(input_ids) >= max_seq_len:
            break

    return {
        "lines": lines,
        "input_ids": input_ids,
        "token_strings": token_strings,
        "token_line": token_line,
        "line_spans": line_spans,
    }


# --------------------------------------------------------------------------- #
# Trust-Hub ingestion (diff-based line labeling)
# --------------------------------------------------------------------------- #
_VERILOG_EXT = (".v", ".sv")
_FREE_KEYS = ("tjfree", "trojanfree", "free", "golden", "clean", "orig", "base")


def _is_verilog(path: str) -> bool:
    return path.lower().endswith(_VERILOG_EXT)


_MAX_RTL_LINES = 6000  # anything larger is a synthesized netlist, not RTL source


def _is_gate_level(path: str) -> bool:
    """Gate-level netlists live under tech folders (180nm/90nm), scan-routed
    files, or *synth* files (e.g. AES aes_synth.v). Our RTL-source pipeline
    skips these."""
    low = path.lower()
    return ("/180nm/" in low or "/90nm/" in low or "\\180nm\\" in low
            or "\\90nm\\" in low or "scan_route" in low or "_syn." in low
            or "synth" in low)


def _is_testbench(name: str) -> bool:
    low = name.lower()
    return low.startswith("test") or low.startswith("tb_") or low.endswith("_tb.v")


def diff_trojan_lines(free_src: str, inf_src: str) -> List[int]:
    """1-indexed lines of ``inf_src`` inserted/replaced vs ``free_src``.
    Whitespace-only differences are ignored."""
    def norm(lines):
        return [re.sub(r"\s+", " ", ln).strip() for ln in lines]

    sm = difflib.SequenceMatcher(
        a=norm(free_src.splitlines()), b=norm(inf_src.splitlines()), autojunk=False
    )
    trojan: List[int] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            trojan.extend(range(j1 + 1, j2 + 1))
    return sorted(set(trojan))


def _find_subdir(bench_dir: str, target: str) -> Optional[str]:
    for dp, _dn, _fn in os.walk(bench_dir):
        if os.path.basename(dp).lower() == target:
            return dp
    return None


def _vfiles(d: str, include_testbench: bool) -> List[str]:
    res = []
    for dp, _dn, fn in os.walk(d):
        for f in fn:
            p = os.path.join(dp, f)
            if not _is_verilog(f) or _is_gate_level(p):
                continue
            if (not include_testbench) and _is_testbench(f):
                continue
            try:  # skip oversized files (synthesized netlists masquerading as .v)
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    if sum(1 for _ in fh) > _MAX_RTL_LINES:
                        continue
            except OSError:
                continue
            res.append(p)
    return sorted(res)


def parse_trusthub(root_dir: str, include_testbench: bool = False) -> List[dict]:
    """Parse Trust-Hub RTL benchmarks into diff-labeled records.

    Handles the native Trust-Hub layout directly: unzip the RS232/AES benchmark
    archives under ``root_dir`` (one folder per benchmark) and run this. For each
    benchmark it:
      * uses the bundled ``TjFree/`` + ``TjIn/`` RTL when present (precise diff),
      * else falls back to a 'free'-named baseline in a flat ``src/`` folder,
      * skips gate-level netlists (180nm/90nm/scan-routed) and (by default)
        testbenches,
      * skips benchmarks that have no golden baseline (to avoid mislabeling).

    Emits records ``{"file": <rel path>, "label": 0/1, "trojan_lines": [...]}``.
    A file that appears in TjIn but is byte-identical to its TjFree twin is a
    genuine clean module and is labeled 0 — giving real negatives for free.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Trust-Hub root not found: {root_dir}")

    benches = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
               if os.path.isdir(os.path.join(root_dir, d))]
    if not benches:
        benches = [root_dir]

    records: List[dict] = []
    skipped: List[str] = []

    for bench in benches:
        name = os.path.basename(bench)
        tjfree = _find_subdir(bench, "tjfree")
        tjin = _find_subdir(bench, "tjin")

        if tjfree and tjin:
            free_map = {os.path.basename(p): p
                        for p in _vfiles(tjfree, include_testbench=True)}
            n_pos = 0
            for inf in _vfiles(tjin, include_testbench):
                base = os.path.basename(inf)
                inf_src = open(inf, encoding="utf-8", errors="ignore").read()
                free = free_map.get(base)
                if free:
                    # in-place Trojan: diff against the clean twin
                    free_src = open(free, encoding="utf-8", errors="ignore").read()
                    lines = diff_trojan_lines(free_src, inf_src)
                    label = 1 if lines else 0
                else:
                    # additive Trojan: file exists only in TjIn (e.g. AES TSC.v,
                    # lfsr.v) -> the whole added module is malicious.
                    lines = list(range(1, len(inf_src.splitlines()) + 1))
                    label = 1
                n_pos += label
                records.append({"file": os.path.relpath(inf, root_dir),
                                "label": label, "trojan_lines": lines})
            for free in _vfiles(tjfree, include_testbench):
                records.append({"file": os.path.relpath(free, root_dir),
                                "label": 0, "trojan_lines": []})
            print(f"[trusthub] {name}: TjFree/TjIn -> {n_pos} trojaned module(s)")
            continue

        # Flat layout: need a 'free'-named baseline to diff against.
        vf = _vfiles(bench, include_testbench)
        if not vf:
            skipped.append(f"{name} (gate-level / no RTL)")
            continue
        frees = [f for f in vf
                 if any(k in os.path.basename(f).lower() for k in _FREE_KEYS)]
        if not frees:
            skipped.append(f"{name} (RTL but no TjFree baseline)")
            continue
        free = frees[0]
        free_src = open(free, encoding="utf-8", errors="ignore").read()
        records.append({"file": os.path.relpath(free, root_dir),
                        "label": 0, "trojan_lines": []})
        for inf in vf:
            if inf == free:
                continue
            lines = diff_trojan_lines(
                free_src, open(inf, encoding="utf-8", errors="ignore").read())
            records.append({"file": os.path.relpath(inf, root_dir),
                            "label": 1 if lines else 0, "trojan_lines": lines})
        print(f"[trusthub] {name}: flat baseline pairing")

    if skipped:
        print("[trusthub] skipped (not usable for RTL diff-labeling):")
        for s in skipped:
            print(f"           - {s}")
    if not records:
        raise FileNotFoundError(
            "No usable RTL benchmarks found. RS232-T2100..T2400 (with bundled "
            "TjFree/TjIn) are the clean set; T1400..T2000 are gate-level; "
            "T200..T901 lack a matching golden baseline.")
    return records


# --------------------------------------------------------------------------- #
# Dataset builder (shared by toy + Trust-Hub)
# --------------------------------------------------------------------------- #
def load_labels(labels_path: str) -> List[dict]:
    records = []
    with open(labels_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _chunk_mapped(mapped, rec, kind, window, overlap):
    """Split a fully-tokenized module into overlapping windows of <= ``window``
    tokens. Each chunk gets its own line map with lines RENUMBERED 1..k (so the
    metrics' 1..n_lines universe stays correct) and the subset of Trojan lines
    that fall inside it. Modules that already fit produce a single chunk with the
    original numbering (so RS232 is unchanged)."""
    ids = mapped["input_ids"]
    tl = mapped["token_line"]
    toks = mapped["token_strings"]
    src = mapped["lines"]
    tset = set(int(x) for x in rec.get("trojan_lines", []))
    T = len(ids)

    if T <= window:
        windows = [(0, T)]
    else:
        step = max(1, window - overlap)
        windows = [(s, min(s + window, T)) for s in range(0, T, step)]
        windows = [w for w in windows if w[0] < w[1]]

    multi = len(windows) > 1
    out = []
    for ci, (s, e) in enumerate(windows):
        w_tl = [int(x) for x in tl[s:e]]
        # source-faithful offset: line numbers stay in source order (blank lines
        # preserved), shifted so the chunk's first line is 1. ``line_offset``
        # recovers the absolute source line: absolute = relative + line_offset.
        lo = min(w_tl)
        hi = max(w_tl)
        offset = lo - 1
        rel = [ln - offset for ln in w_tl]
        spans = {}
        for i, r in enumerate(rel):
            spans.setdefault(str(r), [i, i + 1])[1] = i + 1
        chunk_src = src[lo - 1:hi]                        # real source lines in span
        trojan_rel = sorted(ln - offset for ln in tset if lo <= ln <= hi)
        out.append({
            "file": rec["file"] if not multi else f"{rec['file']}#c{ci}",
            "parent_file": rec["file"],
            "chunk_index": ci,
            "n_chunks": len(windows),
            "line_offset": offset,
            "label": 1 if trojan_rel else 0,
            "trojan_lines": trojan_rel,
            "n_lines": hi - lo + 1,
            "n_tokens": e - s,
            "lines": chunk_src,
            "input_ids": ids[s:e],
            "token_strings": toks[s:e],
            "token_line": rel,
            "line_spans": spans,
            "tokenizer_kind": kind,
        })
    return out


def build_dataset(cfg: dict) -> List[dict]:
    paths = cfg["paths"]
    model_name = cfg["model"]["name"]
    max_seq_len = int(cfg["model"].get("max_seq_len", 2048))
    overlap = int(cfg["model"].get("chunk_overlap", 128))

    tokenizer, kind = build_tokenizer(model_name)
    print(f"[prepare] tokenizer kind = {kind}")

    labels = load_labels(paths["labels"])
    base_dir = paths.get("toy_dir", ".")
    out_records = []
    for rec in labels:
        fpath = os.path.join(base_dir, rec["file"])
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
            source = fh.read()

        # tokenize the WHOLE module (no truncation), then chunk.
        mapped = tokenize_with_line_map(source, tokenizer, kind, max_seq_len=10 ** 9)
        chunks = _chunk_mapped(mapped, rec, kind, max_seq_len, overlap)
        out_records.extend(chunks)
        tag = f" ({len(chunks)} chunks)" if len(chunks) > 1 else ""
        for c in chunks:
            print(f"[prepare] {c['file']}: label={c['label']} "
                  f"lines={c['n_lines']} tokens={c['n_tokens']} "
                  f"trojan_lines={c['trojan_lines']}{tag}")
    return out_records


def write_jsonl(records: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[prepare] wrote {len(records)} records -> {out_path}")


def read_processed(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess RTL into line-aware data")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--trusthub", metavar="DIR",
                    help="parse a Trust-Hub RTL directory into a labels.jsonl "
                         "(diff-based) instead of building the toy dataset")
    ap.add_argument("--write-labels", metavar="PATH",
                    help="where to write the generated labels.jsonl "
                         "(default: <DIR>/labels.jsonl)")
    ap.add_argument("--include-testbench", action="store_true",
                    help="also include test_*/tb_* files (default: skip them)")
    ap.add_argument("--sources", nargs="+", metavar="BASE:LABELS",
                    help="combine multiple 'base_dir:labels.jsonl' pairs into one "
                         "processed dataset (chunked). Use with --out.")
    ap.add_argument("--out", metavar="PATH",
                    help="output processed jsonl for --sources "
                         "(default: data/processed/combined.jsonl)")
    args = ap.parse_args(argv)

    if args.sources:
        cfg = load_config(args.config)
        model_cfg = cfg["model"]
        all_recs = []
        for spec in args.sources:
            base, labels = spec.split(":", 1)
            print(f"\n[combine] source: base={base} labels={labels}")
            sub = {"model": model_cfg,
                   "paths": {"toy_dir": base, "labels": labels}}
            recs = build_dataset(sub)
            all_recs.extend(recs)
        out = args.out or "data/processed/combined.jsonl"
        write_jsonl(all_recs, out)
        n_pos = sum(1 for r in all_recs if int(r["label"]) == 1)
        print(f"[combine] TOTAL {len(all_recs)} records ({n_pos} positive) -> {out}")
        return 0

    if args.trusthub:
        records = parse_trusthub(args.trusthub, include_testbench=args.include_testbench)
        out = args.write_labels or os.path.join(args.trusthub, "labels.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        n_pos = sum(1 for r in records if r["label"] == 1)
        print(f"[trusthub] wrote {len(records)} label records "
              f"({n_pos} trojaned) -> {out}")
        print("[trusthub] Next: in config.yaml set\n"
              f"    toy_dir:   \"{args.trusthub}\"\n"
              f"    labels:    \"{out}\"\n"
              "    processed: \"data/processed/trusthub.jsonl\"\n"
              "then run: python scripts/run_pipeline.py --config config.yaml")
        return 0

    cfg = load_config(args.config)
    records = build_dataset(cfg)
    write_jsonl(records, cfg["paths"]["processed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
