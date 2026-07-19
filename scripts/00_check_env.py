#!/usr/bin/env python3
"""Step 2 — verify the Apple Silicon / MPS environment.

This script is intentionally pure-Python + torch only. It degrades gracefully
if torch is not installed so you get a helpful message instead of a traceback.

It prints:
  * the Python version,
  * the torch version,
  * whether the MPS (Apple GPU) backend is available, and
  * the result of a tiny tensor op executed on MPS (or CPU fallback).
"""
import platform
import sys


def main() -> int:
    print("=" * 60)
    print("TrojanLens environment check")
    print("=" * 60)
    print(f"Python     : {platform.python_version()} ({sys.executable})")
    print(f"Platform   : {platform.platform()}")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - we want any import failure here
        print("torch      : NOT INSTALLED")
        print(f"  -> import failed: {exc}")
        print("  -> Install with:  pip install \"torch>=2.2\"")
        print("     (Apple Silicon default wheels already include the MPS backend.)")
        return 1

    print(f"torch      : {torch.__version__}")

    # MPS availability. `is_built` tells us the wheel has MPS support compiled in;
    # `is_available` tells us this machine can actually use it.
    mps_built = getattr(torch.backends, "mps", None) is not None and \
        torch.backends.mps.is_built()
    mps_avail = getattr(torch.backends, "mps", None) is not None and \
        torch.backends.mps.is_available()
    print(f"MPS built  : {mps_built}")
    print(f"MPS avail  : {mps_avail}")

    if mps_avail:
        device = "mps"
    else:
        device = "cpu"
        print("  -> MPS not available; falling back to CPU.")
        print("     Check: macOS >= 12.3, Apple Silicon, and a non-CUDA torch build.")

    # Tiny tensor op to prove the device works end to end.
    try:
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = (x @ y).sum().item()
        print(f"Tensor op  : ran a 3x3 matmul on '{device}', sum={z:.4f}  OK")
    except Exception as exc:  # noqa: BLE001
        print(f"Tensor op  : FAILED on '{device}': {exc}")
        return 1

    print("-" * 60)
    print(f"Ready. Use device='{device}' in config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
