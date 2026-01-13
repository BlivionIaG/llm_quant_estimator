import argparse
import psutil
from huggingface_hub import HfApi
import re
from typing import Optional

from gpu_utils import get_vram, get_gpu_count
from model_estimator import ModelEstimator




QUANTIZATION_TABLE = {
    "f16": 2.0,
    "fp16": 2.0,
    "f32": 4.0,
    "fp32": 4.0,
    "int8": 1.0,
    "i8": 1.0,
    "int4": 0.5,
    "i4": 0.5,
    "gguf": 0.6,
    "gptq": 0.5,
    "awq": 0.5,
    "bitsandbytes": 0.5,
    "q2_k": 2.56 / 8,
    "q3_k_s": 3.41 / 8,
    "q3_k_m": 3.51 / 8,
    "q3_k_l": 3.82 / 8,
    "q4_0": 4.10 / 8,
    "q4_1": 4.65 / 8,
    "q4_k_s": 4.58 / 8,
    "q4_k_m": 4.80 / 8,
    "q4_k_l": 4.90 / 8, # Approx
    "q5_k_s": 5.54 / 8,
    "q5_k_m": 5.69 / 8,
    "q6_k": 6.59 / 8,
    "q8_0": 8.50 / 8,
    "q8_k": 8.50 / 8, # Approx
    # Fallback prefixes
    "q2": 2.56 / 8,
    "q3": 3.41 / 8,
    "q4": 4.50 / 8, # Average
    "q5": 5.50 / 8,
    "q6": 6.59 / 8,
    "q8": 8.50 / 8,
}

def get_system_ram() -> int:
    """Get total system RAM.
    
    Returns:
        Total RAM in bytes
    """
    return psutil.virtual_memory().total

def format_to_human_readable(bytes_val: int) -> str:
    """Format bytes to human-readable string.
    
    Args:
        bytes_val: Number of bytes
        
    Returns:
        Human-readable string (e.g., "1.23 GB")
    """
    TB = 1024 ** 4
    GB = 1024 ** 3
    MB = 1024 ** 2
    KB = 1024
    
    if bytes_val >= TB:
        return f"{bytes_val / TB:.2f} TB"
    if bytes_val >= GB:
        return f"{bytes_val / GB:.2f} GB"
    if bytes_val >= MB:
        return f"{bytes_val / MB:.2f} MB"
    if bytes_val >= KB:
        return f"{bytes_val / KB:.2f} KB"
    return f"{bytes_val} bytes"

# Regex pattern for GGUF quantization tags
GGUF_QUANT_PATTERN = re.compile(r'(I?Q\d_[A-Z0-9_]+|F16|F32|BF16)', re.IGNORECASE)

def list_available_quants(repo_id: str) -> dict[str, str]:
    """List available quantization variants in a GGUF repository.
    
    Args:
        repo_id: Hugging Face repository ID
        
    Returns:
        Dictionary mapping quantization tags to filenames
    """
    api = HfApi()
    
    try:
        files = api.list_repo_files(repo_id=repo_id)
        available_quants = {}
        
        for file in files:
            if file.lower().endswith(".gguf"):
                match = GGUF_QUANT_PATTERN.search(file)
                if match:
                    quant_tag = match.group(0).upper()
                    available_quants[quant_tag] = file
        
        return available_quants

    except Exception as e:
        print(f"Error accessing repository: {e}")
        return {}

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Estimate recommended quantization for a given LLM model based on system resources")
    parser.add_argument("model_id", help="Hugging Face Model ID (e.g., unsloth/gpt-oss-20b-GGUF)")
    parser.add_argument("--context", type=int, default=4096, help="Context length (seq_len) to account for KV cache (default 4096)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default 1)")
    parser.add_argument("--kv-bits", type=int, default=16, help="KV cache quantization bits (default 16)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Get system resources
    ram = get_system_ram()
    vram = get_vram()
    gpu_count = get_gpu_count()
    
    # Create model estimator
    estimator = ModelEstimator(args.model_id)
    
    recommended = ""
    recommended_size = 0
    
    print(f"System Resources:")
    print(f"  RAM: {format_to_human_readable(ram)}")
    print(f"  VRAM: {format_to_human_readable(vram)} (Total across {gpu_count} GPUs)")
    print(f"  Target Context: {args.context}")
    print(f"  Estimated KV Cache: {format_to_human_readable(estimator.estimate_kv_cache_size(args.context, args.batch, args.kv_bits))}")
    print(f"  Model Size: {estimator.get_parameter_count()/1e9:.2f}B parameters")
    print("-" * 32)

    quants = list_available_quants(args.model_id)

    print(f"Available quants in {args.model_id}:")
    for tag, filename in sorted(quants.items()):
        quantization = tag
        q_key = quantization.lower()
        quant_factor = QUANTIZATION_TABLE.get(q_key)
        
        # Fallback: try prefix (e.g., q4_k_xl -> q4)
        if quant_factor is None:
            prefix = q_key.split("_")[0]
            quant_factor = QUANTIZATION_TABLE.get(prefix, 1.0)
        
        estimated_size = estimator.estimate_quantized_size(quantization, quant_factor, args.context, args.batch, args.kv_bits)
        print(f"Estimated size for {tag}: {format_to_human_readable(estimated_size)}")
        if estimated_size < vram and estimated_size > recommended_size:
            recommended = tag
            recommended_size = estimated_size

    print(f"Recommended quantization: {recommended} ({format_to_human_readable(recommended_size)})")

if __name__ == "__main__":
    main()