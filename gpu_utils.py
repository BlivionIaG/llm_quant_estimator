"""GPU detection utilities for ROCm-based systems."""

import subprocess
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def get_rocm_info() -> tuple[int, int]:
    """Get ROCm GPU information by running rocm-smi once.
    
    Returns:
        Tuple of (total_vram_bytes, gpu_count)
        Returns (0, 0) on error
    """
    rocm_command = "rocm-smi --showmeminfo vram --csv"
    
    result = subprocess.run(rocm_command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}", file=sys.stderr)
        return 0, 0
    
    lines = result.stdout.splitlines()[1:-1]
    vram = sum(int(line.split(",")[1]) for line in lines)
    gpu_count = len(lines)
    
    return vram, gpu_count


def get_vram() -> int:
    """Get total VRAM across all GPUs.
    
    Returns:
        Total VRAM in bytes
    """
    vram, _ = get_rocm_info()
    return vram


def get_gpu_count() -> int:
    """Get the number of GPUs available.
    
    Returns:
        Number of GPUs detected
    """
    _, gpu_count = get_rocm_info()
    return gpu_count
