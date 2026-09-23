# How to run the revision experiments

Two environments. **CPU (M1)** covers three of the four; **Colab (GPU)** covers the
fine-tuned obfuscation and, optionally, the fine-tuned multi-seed.

## A. CPU — one command (family-disjoint, α ablation, multi-seed)
From the artifact root:
```
bash run_revision_experiments.sh
```
Produces, and prints Det F1 / PLC / IoU / VR for:
- `runs/fd_aes2rs232.jsonl`, `runs/fd_rs2322aes.jsonl` → **Table 9 (family-disjoint, M1/M3)**
- `runs/alpha_0.50.jsonl` … `alpha_0.90.jsonl` → **Table 10 (α ablation, R2-5)**
- `runs/seed_13/42/101.jsonl` → **Table 12 (multi-seed, m2)**

New flags added for this: `run_experiment.py --alpha <x> --seed <n>`, and the new
`scripts/12_family_disjoint.py --train-family <F> --test-family <F>`.

## B. Colab (GPU T4) — fine-tuned obfuscation (M4) + fine-tuned timing (R2-7 GPU)
In `TrojanLens_Colab_campaign.ipynb`, after the LoRA fine-tune cell, add one cell:
```python
# 1) generate the obfuscated datasets (data-side; CPU-fine but do it here for one env)
!python scripts/10_obfuscate.py --config config.yaml \
    --sources data/trusthub_rs232:data/trusthub_rs232/labels.jsonl \
              data/trusthub_aes:data/trusthub_aes/labels.jsonl \
    --transforms T1 T2 T3 T123 --max-seq-len 100000 --out-dir data/processed

# 2) evaluate the FINE-TUNED model on each obfuscated set + clean
import time, torch
for tag in ["clean","T1","T2","T3","T123"]:
    path = "data/processed/combined.jsonl" if tag=="clean" else f"data/processed/combined_{tag}.jsonl"
    t0=time.time()
    # eval_finetuned(...) is the notebook's existing fine-tuned eval fn used for Table 5;
    # point it at `path` and record det F1 / PLC / IoU / VR:
    res = eval_finetuned(path)              # <- reuse your existing fine-tuned eval
    print(tag, res, f"{(time.time()-t0):.1f}s")
    if torch.cuda.is_available():
        print("peak GPU MB:", torch.cuda.max_memory_allocated()/1e6); torch.cuda.reset_peak_memory_stats()
```
Report the 5 fine-tuned rows → **Table 8 fine-tuned block (M4)**, and the per-stage
GPU seconds → **Table 11 (R2-7 GPU column)**.

(Optional fine-tuned multi-seed for m2: wrap your LoRA fine-tune + eval in
`for seed in [13,42,101]:`, set `torch.manual_seed(seed)`, and report mean±std.)

## C. Optional stretch — GHOST (M2)
Only if GHOST is easily downloadable; otherwise the response letter's M2 rebuttal stands.
```
python scripts/01_prepare_data.py --trusthub <ghost_dir> --write-labels <ghost>/labels.jsonl
python scripts/run_experiment.py --config config.yaml --processed <ghost>/labels.jsonl --no-verify
```

## What to send back
The printed metric summaries (Det F1 / PLC / IoU / VR) for each run, or the
`runs/*.jsonl` files. I'll drop them into the `[FILL]` cells (Tables 8–12) and the
response-letter brackets, recompile the anonymized Elsevier PDF, and re-verify anonymity.
