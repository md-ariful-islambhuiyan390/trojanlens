#!/usr/bin/env python3
"""Phase-1 obfuscation suite — build semantics-preserving perturbed datasets
for the robustness study (Table 8: Clean / T1 / T2 / T3 / T1+T2+T3).

Reviewer 1, point 2g / M1: the obfuscation column is empty. This produces the
transformed corpora so you can re-run the SAME leave-one-variant-out evaluation
on each and fill the table.

Transforms (each is semantics-preserving and label-safe):
  * T1  identifier renaming      — consistent rename of user identifiers; line
        structure is unchanged, so Trojan line numbers are unchanged.
  * T2  declaration reordering   — reverse each maximal run of consecutive
        wire/reg/port/assign/param lines; Trojan lines are remapped through the
        permutation.
  * T3  dead-code padding        — insert benign `assign _dead_k = 1'b0;` lines;
        Trojan lines are remapped by the number of insertions before them.
  * T123 all three, composed in order T1 -> T2 -> T3.

CRITICAL: line labels are remapped through every transform, so the produced
records carry correct `trojan_lines`. We reuse the *exact* tokenizer and
chunker from 01_prepare_data.py so the output records are byte-for-byte in the
same format your run_experiment.py already consumes.

Usage
-----
    # build clean + all obfuscated corpora next to your real data:
    python scripts/10_obfuscate.py --config config.yaml \
        --sources data/trusthub_rs232:data/trusthub_rs232/labels.jsonl \
                  data/trusthub_aes:data/trusthub_aes/labels.jsonl \
        --transforms none T1 T2 T3 T123 --out-dir data/processed

Then evaluate each (see run_experiment.py --processed override):
    python scripts/run_experiment.py --config config.yaml \
        --processed data/processed/combined_T1.jsonl --no-verify
"""
import argparse
import importlib.util
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PD = load_module("01_prepare_data.py", "prepare_data")

# Verilog keywords we must NOT rename (subset sufficient for Trust-Hub RTL).
_VERILOG_KW = {
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "assign",
    "always", "begin", "end", "if", "else", "case", "endcase", "for", "while",
    "posedge", "negedge", "or", "and", "not", "xor", "nand", "nor", "xnor",
    "parameter", "localparam", "integer", "genvar", "generate", "endgenerate",
    "initial", "function", "endfunction", "task", "endtask", "default", "begin",
    "signed", "unsigned", "real", "time", "logic", "bit", "byte", "int",
    "posedge", "negedge", "repeat", "forever", "wait", "disable", "casex",
    "casez", "defparam", "specify", "endspecify", "primitive", "endprimitive",
    "b", "h", "d", "o",  # base letters in 8'hAB etc.
}

_IDENT = re.compile(r"[A-Za-z_]\w*")
_DECL = re.compile(r"^\s*(wire|reg|input|output|inout|parameter|localparam|assign)\b")


# --------------------------------------------------------------------------- #
# Each transform returns (new_lines, old2new) where old2new maps a 1-indexed
# ORIGINAL line to its 1-indexed line in new_lines (or omits removed lines).
# --------------------------------------------------------------------------- #
def t_identity(lines, seed=0):
    return list(lines), {i: i for i in range(1, len(lines) + 1)}


def t1_rename(lines, seed=0):
    """Consistent identifier rename; line structure preserved (identity map)."""
    rng = random.Random(seed)
    mapping = {}

    def repl(m):
        w = m.group(0)
        if w.lower() in _VERILOG_KW or w[0].isdigit():
            return w
        if w not in mapping:
            mapping[w] = f"{w}_r{rng.randint(100, 999)}"
        return mapping[w]

    new = []
    for ln in lines:
        # protect base-literals like 8'hAB / 1'b0 from identifier renaming
        parts = re.split(r"(\d+'[bBhHoOdD][0-9a-fA-FxXzZ_]+)", ln)
        rebuilt = "".join(p if (i % 2) else _IDENT.sub(repl, p)
                          for i, p in enumerate(parts))
        new.append(rebuilt)
    return new, {i: i for i in range(1, len(lines) + 1)}


def t2_reorder(lines, seed=0):
    """Reverse each maximal run (len>=2) of consecutive declaration lines.
    Trojan lines are remapped through the resulting permutation."""
    n = len(lines)
    perm = list(range(n))  # perm[newpos] = oldpos (0-indexed)
    i = 0
    while i < n:
        if _DECL.match(lines[i]):
            j = i
            while j < n and _DECL.match(lines[j]):
                j += 1
            if j - i >= 2:
                perm[i:j] = list(reversed(perm[i:j]))
            i = j
        else:
            i += 1
    new = [lines[perm[k]] for k in range(n)]
    old2new = {}
    for newpos, oldpos in enumerate(perm):
        old2new[oldpos + 1] = newpos + 1
    return new, old2new


def t3_pad(lines, every=4, seed=0):
    """Insert a benign dead-code line after every `every`-th source line.
    Dead-code assigns an otherwise-unused net, so it is semantics-preserving."""
    new = []
    old2new = {}
    pad = 0
    for idx, ln in enumerate(lines, start=1):
        new.append(ln)
        old2new[idx] = len(new)  # new 1-indexed position of this original line
        if idx % every == 0:
            new.append(f"    wire _dead_{pad}; assign _dead_{pad} = 1'b0;")
            pad += 1
    return new, old2new


def compose(lines, transforms, seed=0):
    """Apply transforms left-to-right, composing the line maps."""
    cur = list(lines)
    cmap = {i: i for i in range(1, len(lines) + 1)}
    for k, t in enumerate(transforms):
        cur, m = t(cur, seed=seed + k)
        cmap = {orig: m[newpos] for orig, newpos in cmap.items() if newpos in m}
    return cur, cmap


_REGISTRY = {
    "none": [t_identity],
    "T1": [t1_rename],
    "T2": [t2_reorder],
    "T3": [t3_pad],
    "T123": [t1_rename, t2_reorder, t3_pad],
}


def remap_trojan(trojan_lines, old2new):
    return sorted(old2new[l] for l in trojan_lines if l in old2new)


def build_transformed(cfg, sources, transform_name, seed=1234, max_seq_len=None):
    """Return processed (chunked) records for one transform, reusing the exact
    tokenizer + chunker from 01_prepare_data.py.

    ``max_seq_len`` overrides cfg['model']['max_seq_len']. Set it large (e.g.
    100000) so each module stays ONE record: identifier-rename and padding
    inflate token counts, and if that pushes a module past the chunk window it
    would split into extra chunks, changing the number of evaluated units per
    transform and making the robustness comparison unfair. Keeping one record
    per module holds the unit count constant (469 modules / 95 positive) across
    all transforms."""
    model_cfg = cfg["model"]
    if max_seq_len is None:
        max_seq_len = int(model_cfg.get("max_seq_len", 2048))
    overlap = int(model_cfg.get("chunk_overlap", 128))
    tokenizer, kind = PD.build_tokenizer(model_cfg["name"])
    print(f"[obf:{transform_name}] tokenizer kind = {kind}")
    transforms = _REGISTRY[transform_name]

    out = []
    for spec in sources:
        base, labels_path = spec.split(":", 1)
        for rec in PD.load_labels(labels_path):
            fpath = os.path.join(base, rec["file"])
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                lines = fh.read().splitlines()
            new_lines, old2new = compose(lines, transforms, seed=seed)
            new_trojan = remap_trojan([int(x) for x in rec.get("trojan_lines", [])],
                                      old2new)
            rec2 = {"file": rec["file"], "label": 1 if new_trojan else 0,
                    "trojan_lines": new_trojan}
            source = "\n".join(new_lines)
            mapped = PD.tokenize_with_line_map(source, tokenizer, kind,
                                               max_seq_len=10 ** 9)
            out.extend(PD._chunk_mapped(mapped, rec2, kind, max_seq_len, overlap))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build obfuscated datasets")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sources", nargs="+", required=True, metavar="BASE:LABELS")
    ap.add_argument("--transforms", nargs="+",
                    default=["none", "T1", "T2", "T3", "T123"],
                    choices=list(_REGISTRY))
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-seq-len", type=int, default=100000,
                    help="chunk window (default 100000 = never split, so every "
                         "module is ONE record and the unit count stays constant "
                         "across transforms). Lower it only if you hit memory limits.")
    args = ap.parse_args(argv)

    cfg = PD.load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    baseline_pos = None
    for t in args.transforms:
        recs = build_transformed(cfg, args.sources, t, seed=args.seed,
                                 max_seq_len=args.max_seq_len)
        suffix = "clean" if t == "none" else t
        out = os.path.join(args.out_dir, f"combined_{suffix}.jsonl")
        PD.write_jsonl(recs, out)
        n_pos = sum(1 for r in recs if int(r["label"]) == 1)
        if baseline_pos is None:
            baseline_pos = (len(recs), n_pos)
        flag = "" if (len(recs), n_pos) == baseline_pos else \
            "  <-- WARNING: unit count differs from baseline; raise --max-seq-len"
        print(f"[obf:{t}] {len(recs)} records ({n_pos} positive) -> {out}{flag}\n")
    print("[obf] Done. Evaluate each with:\n"
          "  python scripts/run_experiment.py --config config.yaml "
          "--processed <file> --no-verify   # Det F1 + PLC (fast)\n"
          "Add verify (VR column) only on GPU — it is ~minutes/positive on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
