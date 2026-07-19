#!/usr/bin/env python3
"""TrojanLens model + core mechanisms.

Paper concept
-------------
A single RTL-native backbone (Qwen2.5-Coder-1.5B, LoRA-adapted) feeds two heads:

  * a pooled **binary detection** head  -> "is there a Trojan in this module?"
  * a per-token **localization** head    -> "which lines are the Trojan?"
    (per-token logits are aggregated to source lines via the line<->token map)

The same module also provides the two mechanisms the verification layer needs:

  * :func:`focal_loss`      — class-imbalance-robust detection loss.
  * :func:`neutralize_lines`— replace the tokens of chosen source lines with a
    neutral (pad) token; used for the counterfactual "does removing the cited
    lines flip the decision?" check.

Heavy imports (torch / transformers / peft) are guarded so this file imports
(and ``py_compile``-checks) even where those libraries are absent. A lightweight
``toy`` encoder lets you exercise all shapes/logic offline without downloading
Qwen weights.
"""
from typing import Dict, List, Optional, Sequence, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except Exception:  # noqa: BLE001
    _TORCH = False


# --------------------------------------------------------------------------- #
# Pure-python utility: neutralize (mask) whole source lines.
# Works on plain lists so it is usable even without torch (e.g. in tests).
# --------------------------------------------------------------------------- #
def neutralize_lines(
    input_ids: Sequence[int],
    token_line: Sequence[int],
    lines_to_mask: Sequence[int],
    mask_id: int = 0,
) -> List[int]:
    """Return a copy of ``input_ids`` with every token that belongs to a line
    in ``lines_to_mask`` replaced by ``mask_id`` (a neutral / pad token).

    ``token_line[i]`` is the 1-indexed source line of token ``input_ids[i]``.
    This is the counterfactual "surgery" used by verification: removing the
    Trojan's tokens should make the design look clean.
    """
    if _TORCH and hasattr(input_ids, "tolist"):
        ids = list(input_ids.tolist())
        tl = list(token_line.tolist()) if hasattr(token_line, "tolist") else list(token_line)
    else:
        ids = list(input_ids)
        tl = list(token_line)

    mask_set = set(int(x) for x in lines_to_mask)
    out = [mask_id if int(tl[i]) in mask_set else ids[i] for i in range(len(ids))]
    return out


def aggregate_tokens_to_lines(
    token_values: Sequence[float],
    token_line: Sequence[int],
    reduce: str = "mean",
) -> Dict[int, float]:
    """Aggregate per-token values up to per-line values via the line map.

    Used to turn per-token localization logits *or* per-token attributions into
    per-line scores. ``reduce`` is 'mean' or 'sum'.
    """
    buckets: Dict[int, List[float]] = {}
    for val, ln in zip(token_values, token_line):
        buckets.setdefault(int(ln), []).append(float(val))
    out: Dict[int, float] = {}
    for ln, vals in buckets.items():
        if not vals:
            continue
        out[ln] = sum(vals) / len(vals) if reduce == "mean" else sum(vals)
    return out


# --------------------------------------------------------------------------- #
# torch-dependent parts
# --------------------------------------------------------------------------- #
if _TORCH:

    def focal_loss(
        logits: "torch.Tensor",
        targets: "torch.Tensor",
        gamma: float = 2.0,
        alpha: float = 0.75,
    ) -> "torch.Tensor":
        """Binary focal loss over 2-class detection logits.

        logits  : [B, 2]
        targets : [B]  in {0, 1}
        alpha weights the positive (Trojan) class; (1-alpha) the negative.
        """
        ce = F.cross_entropy(logits, targets, reduction="none")   # [B]
        pt = torch.exp(-ce)                                        # prob of true class
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(ce, alpha),
            torch.full_like(ce, 1.0 - alpha),
        )
        loss = alpha_t * (1.0 - pt) ** gamma * ce
        return loss.mean()

    class _ToyEncoder(nn.Module):
        """A tiny stand-in for the Qwen encoder so shapes/logic run offline.

        Produces a per-token hidden state of size ``hidden``. IDs are taken mod
        ``vocab`` so the fallback tokenizer's arbitrary ids stay in range.
        """

        def __init__(self, vocab: int = 2048, hidden: int = 64):
            super().__init__()
            self.vocab = vocab
            self.hidden = hidden
            self.embed = nn.Embedding(vocab, hidden)
            self.proj = nn.Linear(hidden, hidden)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, attention_mask=None):
            ids = input_ids % self.vocab
            h = self.embed(ids)
            h = torch.tanh(self.proj(h))
            return h  # [B, T, H]

    class TrojanLensModel(nn.Module):
        """Qwen encoder (LoRA) + detection head + per-token localization head.

        forward(input_ids, attention_mask) -> dict:
            det_logits : [B, 2]        module-level Trojan detection
            tok_logits : [B, T]        per-token localization logit
        Aggregate ``tok_logits`` to lines with :func:`aggregate_tokens_to_lines`
        using the record's ``token_line``.
        """

        def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
            lora_cfg: Optional[dict] = None,
            toy: bool = False,
            dtype: Optional[str] = None,
        ):
            super().__init__()
            self.toy = toy
            self.model_name = model_name

            if toy:
                self.encoder = _ToyEncoder()
                hidden = self.encoder.hidden
            else:
                hidden = self._build_real_encoder(model_name, lora_cfg, dtype)

            self.hidden = hidden
            self.detect_head = nn.Linear(hidden, 2)   # pooled -> {clean, trojan}
            self.locate_head = nn.Linear(hidden, 1)   # per-token -> trojan-ness

        # -- real backbone construction (Qwen + LoRA) -- #
        def _build_real_encoder(self, model_name, lora_cfg, dtype) -> int:
            from transformers import AutoModel
            torch_dtype = None
            if dtype in ("bfloat16", "bf16"):
                torch_dtype = torch.bfloat16
            elif dtype in ("float16", "fp16"):
                torch_dtype = torch.float16

            # low_cpu_mem_usage avoids a transient 2x copy during load;
            # transformers >=5 renamed torch_dtype -> dtype, so try dtype first.
            load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
            if torch_dtype is not None:
                try:
                    encoder = AutoModel.from_pretrained(
                        model_name, dtype=torch_dtype, **load_kwargs
                    )
                except TypeError:
                    encoder = AutoModel.from_pretrained(
                        model_name, torch_dtype=torch_dtype, **load_kwargs
                    )
            else:
                encoder = AutoModel.from_pretrained(model_name, **load_kwargs)

            # Attach LoRA adapters (peft). NO bitsandbytes / QLoRA on MPS.
            if lora_cfg is not None:
                from peft import LoraConfig, get_peft_model
                peft_cfg = LoraConfig(
                    r=int(lora_cfg.get("r", 16)),
                    lora_alpha=int(lora_cfg.get("alpha", 32)),
                    lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                    target_modules=list(lora_cfg.get("target_modules", [])),
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                encoder = get_peft_model(encoder, peft_cfg)

            self.encoder = encoder
            hidden = getattr(encoder.config, "hidden_size", None)
            if hidden is None:
                # peft-wrapped models expose config on .base_model
                hidden = encoder.base_model.config.hidden_size
            return int(hidden)

        # -- embedding layer, needed by LayerIntegratedGradients -- #
        def embedding_layer(self):
            if self.toy:
                return self.encoder.get_input_embeddings()
            enc = self.encoder
            if hasattr(enc, "get_input_embeddings"):
                return enc.get_input_embeddings()
            return enc.base_model.get_input_embeddings()

        def _encode(self, input_ids, attention_mask=None):
            if self.toy:
                return self.encoder(input_ids, attention_mask)
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # AutoModel returns last_hidden_state; peft passes it through.
            return out.last_hidden_state

        def forward(self, input_ids, attention_mask=None):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                if attention_mask is not None and attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

            hidden = self._encode(input_ids, attention_mask)          # [B, T, H]

            # Masked mean-pool for detection.
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).to(hidden.dtype)     # [B, T, 1]
                summed = (hidden * m).sum(dim=1)
                counts = m.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            else:
                pooled = hidden.mean(dim=1)

            det_logits = self.detect_head(pooled.float())             # [B, 2]
            tok_logits = self.locate_head(hidden.float()).squeeze(-1)  # [B, T]
            return {"det_logits": det_logits, "tok_logits": tok_logits}

        # Convenience: detection probability of the Trojan class.
        def trojan_prob(self, input_ids, attention_mask=None) -> float:
            out = self.forward(input_ids, attention_mask)
            p = torch.softmax(out["det_logits"], dim=-1)[0, 1]
            return float(p.detach().cpu())

else:  # torch not available -----------------------------------------------

    def focal_loss(*_args, **_kwargs):  # type: ignore
        raise ImportError("focal_loss requires torch. pip install 'torch>=2.2'")

    class TrojanLensModel:  # type: ignore
        """Placeholder so the module imports without torch. Instantiating it
        raises a clear error telling you to install torch."""

        def __init__(self, *_args, **_kwargs):
            raise ImportError(
                "TrojanLensModel requires torch. pip install 'torch>=2.2' "
                "(Apple Silicon wheels include the MPS backend)."
            )


def build_model(cfg: dict, toy: bool = False):
    """Factory used by the training / pipeline scripts."""
    return TrojanLensModel(
        model_name=cfg["model"]["name"],
        lora_cfg=cfg.get("lora"),
        toy=toy,
        dtype=cfg["model"].get("dtype"),
    )


if __name__ == "__main__":
    # Tiny self-test of the toy path (only runs if torch is installed).
    if not _TORCH:
        print("torch not installed; model self-test skipped.")
        raise SystemExit(0)
    cfg = {"model": {"name": "toy", "dtype": "float32"}, "lora": None}
    m = build_model(cfg, toy=True)
    ids = torch.randint(0, 500, (1, 12))
    out = m(ids)
    print("det_logits", tuple(out["det_logits"].shape),
          "tok_logits", tuple(out["tok_logits"].shape))
    tl = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    masked = neutralize_lines(ids[0], tl, [2, 3], mask_id=0)
    print("neutralized len", len(masked), "sample", masked[:8])
