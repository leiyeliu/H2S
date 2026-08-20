#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    # An explicit architecture list is sufficient for a toolkit-only build
    # host, so enable the existing FORCE_CUDA hooks unless the caller opted out.
    export FORCE_CUDA="${FORCE_CUDA:-1}"
    echo "Building CUDA extensions for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
else
    export FORCE_CUDA="${FORCE_CUDA:-0}"
    echo "TORCH_CUDA_ARCH_LIST is unset; PyTorch will infer targets from visible GPUs."
    echo "For a GPU-less build host, set TORCH_CUDA_ARCH_LIST explicitly."
fi

install_extension() {
    local label="$1"
    local source_dir="$2"

    echo "Installing ${label} from ${source_dir}"
    "${PYTHON_BIN}" -m pip install -v --no-build-isolation "${source_dir}"
}

install_extension \
    "MultiScaleDeformableAttention" \
    "${PROJECT_ROOT}/mask2former/modeling/pixel_decoder/ops"
install_extension "smm_cuda" "${PROJECT_ROOT}/third_party/ops_smm"
install_extension \
    "selective_scan" \
    "${PROJECT_ROOT}/third_party/selective_scan"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/check_install.py"
