import argparse
import psutil
import subprocess
from huggingface_hub import hf_hub_download, model_info, HfApi
import json
import re

QUANTIZATION_TABLE = {
    "f16": 2.0,
    "f32": 4.0,
    "int8": 1.0,
    "int4": 0.5,
    "gguf": 0.6, # TODO: calculate
    "gptq": 0.5,
    "awq": 0.5,
    "bitsandbytes": 0.5,
    "q2": float(2/8),
    "q3": float(3/8),
    "q4": float(4/8),
    "q5": float(5/8),
    "q6": float(6/8),
    "q7": float(7/8),
    "q8": float(8/8),
}

ESTIMATED_OVERHEAD = 1.2

def get_system_ram() -> int:
    return psutil.virtual_memory().total

def get_rocm_vram() -> int:
    rocm_command = "rocm-smi --showmeminfo vram --csv"

    result = subprocess.run(rocm_command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}", file=sys.stderr)
        return None
    
    # sum for each card
    vram = 0
    for line in result.stdout.splitlines()[1:-1]:
        vram += int(line.split(",")[1])

    return vram

def get_rocm_gpu_count() -> int:
    rocm_command = "rocm-smi --showmeminfo vram --csv"

    result = subprocess.run(rocm_command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}", file=sys.stderr)
        return None
    
    return len(result.stdout.splitlines()) - 2

def get_rocm_llamacpp_vram(docker: bool = False, docker_engine: str = "docker", rocm_image: str = "rocm/llama.cpp", rocm_image_tag: str = "latest") -> int:
    llamacpp_command = f"{docker_engine} run --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --rm {rocm_image}:{rocm_image_tag} -r" if docker else "llama-cli"

    command = f"{llamacpp_command} --list-devices 2>&1 | sed -n '/Available devices:/,$p' | tail -n +2"
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}", file=sys.stderr)
        return None
    
    # Convert from MiB to bytes and sum
    vram = 0
    for line in result.stdout.splitlines():
        vram += int(line.split(":")[1].split("(")[1].split("MiB")[0]) * 1024 * 1024
    
    return vram

def get_gpu_count() -> int:
    return get_rocm_gpu_count()

def get_vram() -> int:
    # TODO: implement llama.cpp vram detection
    # return get_rocm_llamacpp_vram(docker=True, docker_engine="podman", rocm_image="rocm/llama.cpp", rocm_image_tag="llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_full")
    return get_rocm_vram()

def format_to_human_readable(bytes_val: int) -> str:
    if bytes_val >= 1024**4:
        return f"{bytes_val / (1024**4):.2f} TB"
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    if bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    if bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} bytes"

def get_model_info(model_id: str) -> dict:
    # Get model info from huggingface
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def get_model_size(model_config: dict) -> int:
    # 1. Embeddings (The part you already have)
    # Note: GPT-OSS usually does not tie word embeddings, so we count them twice (input + output)
    embeddings = 2 * (model_config["vocab_size"] * model_config["hidden_size"])

    # 2. Attention Block (Per Layer)
    # Includes Q, K, V projections and the Output projection
    # For GQA: (num_heads + 2 * num_kv_heads) * (head_dim * hidden_size) + (hidden_size^2)
    attn_per_layer = (model_config["num_attention_heads"] + 2 * model_config["num_key_value_heads"]) * \
                    (64 * model_config["hidden_size"]) + (model_config["hidden_size"]**2)

    # 3. MoE Experts (The "Big" Part - Per Layer)
    # Each expert is a SwiGLU MLP: 3 matrices of (hidden_size * intermediate_size)
    expert_size = 3 * (model_config["hidden_size"] * model_config["intermediate_size"])
    all_experts_per_layer = model_config["num_local_experts"] * expert_size

    # 4. Sum it all up
    total_params = embeddings + model_config["num_hidden_layers"] * (attn_per_layer + all_experts_per_layer)
    return total_params

def get_list_of_model_quantizations(model_id: str) -> list:
    from huggingface_hub import HfApi

    api = HfApi()
    base_model_id = model_id

    # Search for models that reference this base model
    quantization_filters = [  
    "f16", "fp16",      # 16-bit float  
    "f32", "fp32",      # 32-bit float    
    "int8", "i8",       # 8-bit integer  
    "int4", "i4",       # 4-bit integer  
    "gguf",             # GGUF format  
    "gptq",             # GPTQ quantization  
    "awq",              # AWQ quantization  
    "bitsandbytes",     # BitsAndBytes quantization  
    ]  
    
    model_types = []
    # Get models with specific quantization  
    for quant in quantization_filters:  
        models = list(api.list_models(filter=quant, limit=5))  
        # print(f"{quant}: {len(models)} models")
        if quant not in model_types:
            model_types.append(quant)

    # print(f"Quantized versions of {base_model_id}:")
    for model in models:
        # Filter for common quantization tags or suffixes
        if any(q in model.id.upper() for q in ["GGUF", "AWQ", "GPTQ", "EXL2", "ONNX"]):
            print(model)
            print(f"- {model.id} ({model.downloads} downloads)")

    # for a given quantization, get the model size
    for model in models:
        if any(q in model.id.upper() for q in ["GGUF", "AWQ", "GPTQ", "EXL2", "ONNX"]):
            print(f"- {model.id} ({model.downloads} downloads)")

    return model_types

def estimate_quantization(model_id: str, quantization: str, context: int, batch: int) -> float:
    model_info = get_model_info(model_id)
    model_size = get_model_size(model_info)
    
    quant_factor: float = QUANTIZATION_TABLE.get(quantization.lower(), 1.0)

    estimated_kv_cache = context * batch * model_info["hidden_size"]
    estimated_size = model_size * quant_factor * ESTIMATED_OVERHEAD + estimated_kv_cache
    
    return estimated_size

def list_available_quants(repo_id):
    api = HfApi()
    
    try:
        # 1. Get all files in the repository
        files = api.list_repo_files(repo_id=repo_id)
        
        # 2. Filter for GGUF files and extract the quant pattern
        # This regex looks for common patterns like Q4_K_M, IQ3_S, etc.
        quant_pattern = re.compile(r'(I?Q\d_[A-Z0-9_]+|F16|F32|BF16)', re.IGNORECASE)
        
        available_quants = {}
        
        for file in files:
            if file.lower().endswith(".gguf"):
                match = quant_pattern.search(file)
                if match:
                    quant_tag = match.group(0).upper()
                    available_quants[quant_tag] = file
        
        return available_quants

    except Exception as e:
        print(f"Error accessing repository: {e}")
        return {}

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate recommended quantization for a given LLM model based on system resources")
    parser.add_argument("model_id", help="Hugging Face Model ID (e.g., openai/gpt-oss-20b)")
    parser.add_argument("--context", type=int, default=4096, help="Context length (seq_len) to account for KV cache (default 4096)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default 1)")

    args = parser.parse_args()

    # Get system resources
    ram = get_system_ram()
    vram = get_vram()
    gpu_count = get_gpu_count()
    model_info = get_model_info(args.model_id)
    model_quantizations = get_list_of_model_quantizations(args.model_id)
    recommended = ""
    recommended_size = 0

    print(f"System Resources:")
    print(f"  RAM: {format_to_human_readable(ram)}")
    print(f"  VRAM: {format_to_human_readable(vram)} (Total across {gpu_count} GPUs)")
    print(f"  Target Context: {args.context}")
    print(f"  Model Size: {get_model_size(model_info)/1e9:.2f}B parameters")
    print("-" * 32)

    for quantization in model_quantizations:
        # print(f"Quantization: {quantization}")
        estimated_size = estimate_quantization(args.model_id, quantization, args.context, args.batch)
        print(f"Estimated size for {quantization}: {format_to_human_readable(estimated_size)}")
        if estimated_size < vram and estimated_size > recommended_size:
            recommended = quantization
            recommended_size = estimated_size

    quants = list_available_quants("unsloth/gpt-oss-20b-GGUF")

    print(f"Available quants in {args.model_id}:")
    for tag, filename in sorted(quants.items()):
        # print(f"  - {tag:<10} (File: {filename})")
        quantization = f"{tag:<10}".split("_")[0]
        estimated_size = estimate_quantization(args.model_id, quantization, args.context, args.batch)
        print(f"Estimated size for {quantization}: {format_to_human_readable(estimated_size)}")
        if estimated_size < vram and estimated_size > recommended_size:
            recommended = quantization
            recommended_size = estimated_size

    print(f"Recommended quantization: {recommended} ({format_to_human_readable(recommended_size)})")

if __name__ == "__main__":
    main()