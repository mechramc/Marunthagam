"""
Side-effect import: register the nvidia/* CUDA runtime DLL directories so that
the prebuilt cu124 llama-cpp-python wheel can find cudart64_12, cublas64_12,
and nvrtc64_120 on a Windows host whose system CUDA toolkit is 13.x.

Import this module BEFORE `import llama_cpp` (or anything that imports it).
On non-Windows hosts and Linux this is a no-op.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        import nvidia
    except ImportError:
        nvidia = None

    if nvidia is not None:
        for nvidia_root in (Path(p) for p in nvidia.__path__):
            for sub in ("cuda_runtime/bin", "cublas/bin", "cuda_nvrtc/bin"):
                dll_dir = nvidia_root / sub
                if dll_dir.is_dir():
                    os.add_dll_directory(str(dll_dir))
