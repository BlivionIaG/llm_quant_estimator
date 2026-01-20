"""GPU detection utilities for ROCm-based systems."""

import subprocess
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def get_rocm_info(debug: bool = False) -> tuple[int, int]:
    """Get ROCm GPU information by running rocm-smi once.

    Returns:
        Tuple of (total_vram_bytes, gpu_count)
        Returns (0, 0) on error
    """
    rocm_command = "rocm-smi --showmeminfo vram --csv"

    result = subprocess.run(rocm_command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        if debug:
            print(f"Error running command: {result.stderr}", file=sys.stderr)
        return 0, 0

    lines = result.stdout.splitlines()[1:-1]
    vram = sum(int(line.split(",")[1]) for line in lines)
    gpu_count = len(lines)

    return vram, gpu_count

@lru_cache(maxsize=1)
def get_nvidia_info() -> tuple[int, int]:
    """Get nvidia GPU information by running nvidia-smi once.

    Returns:
        Tuple of (total_vram_bytes, gpu_count)
        Returns (0, 0) on error
    """
    command = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        if debug:
            print(f"Error running command: {result.stderr}", file=sys.stderr)
        return 0, 0
    lines = result.stdout.splitlines()
    vram = sum(int(line)*1024*1024 for line in lines)
    gpu_count = len(lines)

    return vram, gpu_count

def get_vram() -> int:
    """Get total VRAM across all GPUs.

    Returns:
        Total VRAM in bytes
    """
    vram = sum(f()[0] for f in (get_rocm_info, get_nvidia_info))

    return vram


def get_gpu_count() -> int:
    """Get the number of GPUs available.

    Returns:
        Number of GPUs detected
    """
    gpu_count = sum(f()[1] for f in (get_rocm_info, get_nvidia_info))

    return gpu_count
