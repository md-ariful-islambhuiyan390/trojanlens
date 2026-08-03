# TrojanLens

Code and line-level annotations for **“TrojanLens: A Fine-Tuned RTL Security LLM
with Faithfulness-Verified Localization for Third-Party IP Trojan Auditing.”**

TrojanLens detects hardware Trojans in Verilog RTL, localizes the malicious logic
to specific source lines, produces a natural-language explanation, and —
critically — **mechanically verifies** that explanation (attribution grounding +
counterfactual masking) before surfacing it. This repository contains everything
needed to reproduce the results on the public Trust-Hub benchmarks.

## What’s here

```
scripts/                 pipeline (data prep, model, attribution, verification, metrics, experiment)
  01_prepare_data.py     Trust-Hub diff-labeling + sliding-window chunking
  model.py               Qwen2.5-Coder backbone + detection & localization heads
  05_attributions.py     integrated-gradients / grad×input attributions
  06_verify.py           faithfulness-verification layer
  07_metrics.py          detection / localization / verification metrics
  run_experiment.py      leave-one-variant-out CV (--processed / --no-verify options)
  08_reconcile_counts.py corpus vs. prediction count reconciliation (integrity check)
  09_efficiency.py       per-design latency + peak-memory measurement
  10_obfuscate.py        semantics-preserving obfuscation suite (T1/T2/T3) + label remap
config.yaml              configuration
requirements.txt         Python dependencies
data/toy/                tiny self-contained RS232-style example (runs offline)
TrojanLens_Colab_clean.ipynb      single-split GPU fine-tune + verified evaluation
TrojanLens_Colab_campaign.ipynb   30-fold LOO fine-tune, threshold/IG sensitivity, baselines
```

## Data (Trust-Hub — obtain separately)

We do **not** redistribute the Trust-Hub Verilog (it has its own registration and
terms). To reproduce:

1. Register at <https://trust-hub.org> and download the **RTL** RS232 and AES
   chip-level Trojan benchmarks.
2. Unzip them under `data/trusthub_rs232/` and `data/trusthub_aes/` (one folder
   per benchmark).
3. Generate line-level labels (diff-based; additive-Trojan aware):

   ```bash
   python scripts/01_prepare_data.py --trusthub data/trusthub_rs232 \
       --write-labels data/trusthub_rs232/labels.jsonl
   python scripts/01_prepare_data.py --trusthub data/trusthub_aes \
       --write-labels data/trusthub_aes/labels.jsonl
   ```
4. Build the combined, chunked dataset:

   ```bash
   python scripts/01_prepare_data.py --config config.yaml \
       --sources data/trusthub_rs232:data/trusthub_rs232/labels.jsonl \
                 data/trusthub_aes:data/trusthub_aes/labels.jsonl \
       --out data/processed/combined.jsonl
   ```

The `labels.jsonl` files (our contribution — line-level Trojan annotations) can be
committed here; the raw Verilog should not.

## Reproduce

- **Local (CPU, frozen-feature 30-fold LOO):**
  ```bash
  python scripts/run_experiment.py --config config.yaml
  ```
- **GPU (LoRA fine-tune + verified eval + baselines):** open
  `TrojanLens_Colab_clean.ipynb` in Google Colab (T4), upload
  `data/processed/combined.jsonl`, and run top to bottom.

## Environment

Python 3.11, PyTorch (MPS on Apple Silicon / CUDA on GPU). Install with
`pip install -r requirements.txt`. No bitsandbytes/QLoRA (unsupported on Apple
Silicon); LoRA runs in bf16.

## Citation

If you use this code or the annotations, please cite the paper (see
`CITATION.cff`) and the Trust-Hub benchmarks (Salmani et al., ICCD 2013; Shakya et
al., JHSS 2017).

## License

Code is released under the MIT License (`LICENSE`). Trust-Hub benchmark data is
governed by its own terms and is not included here.
