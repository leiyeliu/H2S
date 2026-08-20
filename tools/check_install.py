#!/usr/bin/env python3
"""Check that H2S's required CUDA extension modules can be imported."""

from __future__ import annotations

import importlib
import sys

try:
    # Loading PyTorch first makes libtorch shared libraries available to the
    # extension modules on environments that do not embed an ELF rpath.
    import torch  # noqa: F401
except Exception as exc:
    print(f"PyTorch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc


REQUIRED_EXTENSIONS = {
    "MultiScaleDeformableAttention": (
        "ms_deform_attn_forward",
        "ms_deform_attn_backward",
    ),
    "smm_cuda": (
        "SMM_QmK_forward_cuda",
        "SMM_QmK_backward_cuda",
        "SMM_AmV_forward_cuda",
        "SMM_AmV_backward_cuda",
    ),
    "selective_scan_cuda_core": ("fwd", "bwd"),
}


def main() -> int:
    failures: list[str] = []

    for module_name, required_symbols in REQUIRED_EXTENSIONS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # Import can fail on ABI/linker errors too.
            failures.append(f"{module_name}: import failed: {exc}")
            continue

        missing = [name for name in required_symbols if not hasattr(module, name)]
        if missing:
            failures.append(
                f"{module_name}: missing symbols: {', '.join(sorted(missing))}"
            )
            continue

        module_path = getattr(module, "__file__", "<built-in>")
        print(f"[ok] {module_name}: {module_path}")

    if failures:
        print("\nCUDA extension check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nRebuild with scripts/install_ops.sh. If needed, set "
            "TORCH_CUDA_ARCH_LIST for the target GPU.",
            file=sys.stderr,
        )
        return 1

    print("All required CUDA extensions are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
