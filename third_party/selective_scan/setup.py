# Modified by Mzero #20240123
# Modified for H2S: remove hard-coded GPU architectures so PyTorch honors
# TORCH_CUDA_ARCH_LIST, and exclude the unused split kernel mode.
# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import subprocess
from pathlib import Path

import torch
from packaging.version import Version, parse
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CUDAExtension,
)
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


MODES = ["core", "ndstate", "oflex"]
# MODES = ["core", "ndstate", "oflex", "nrow"]


def get_ext():
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    print("\n\nCUDA_HOME = {}\n\n".format(CUDA_HOME))

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is not set. Install a CUDA toolkit and make nvcc available."
        )

    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.6"):
        raise RuntimeError(
            "package is only supported on CUDA 11.6 and above.  "
            "Note: make sure nvcc has a supported version by running nvcc -V."
        )

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    sources = dict(
        core=[
            "csrc/selective_scan/cus/selective_scan.cpp",
            "csrc/selective_scan/cus/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cus/selective_scan_core_bwd.cu",
        ],
        nrow=[
            "csrc/selective_scan/cusnrow/selective_scan_nrow.cpp",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd2.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd3.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_fwd4.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd2.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd3.cu",
            "csrc/selective_scan/cusnrow/selective_scan_core_bwd4.cu",
        ],
        ndstate=[
            "csrc/selective_scan/cusndstate/selective_scan_ndstate.cpp",
            "csrc/selective_scan/cusndstate/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusndstate/selective_scan_core_bwd.cu",
        ],
        oflex=[
            "csrc/selective_scan/cusoflex/selective_scan_oflex.cpp",
            "csrc/selective_scan/cusoflex/selective_scan_core_fwd.cu",
            "csrc/selective_scan/cusoflex/selective_scan_core_bwd.cu",
        ],
    )
    names = dict(
        core="selective_scan_cuda_core",
        nrow="selective_scan_cuda_nrow",
        ndstate="selective_scan_cuda_ndstate",
        oflex="selective_scan_cuda_oflex",
    )

    ext_modules = [
        CUDAExtension(
            name=names.get(MODE, None),
            sources=sources.get(MODE, None),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    # PyTorch supplies -gencode flags from TORCH_CUDA_ARCH_LIST,
                    # or detects the visible GPUs when that variable is unset.
                    "--threads",
                    "4",
                ],
            },
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
        )
        for MODE in MODES
    ]

    return ext_modules


ext_modules = get_ext()
setup(
    name="selective_scan",
    version="0.0.2",
    packages=[],
    author="Tri Dao, Albert Gu, Mzero",
    author_email="tri@tridao.me, agu@cs.cmu.edu, liuyue171@mails.ucas.ac.cn",
    description="selective scan",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": _bdist_wheel,
    },
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
    ],
)
