import argparse
import psutil
import subprocess
from huggingface_hub import hf_hub_download, model_info, HfApi
import json

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
    
    # Get models with specific quantization  
    for quant in quantization_filters:  
        models = list(api.list_models(filter=quant, limit=5))  
        print(f"{quant}: {len(models)} models")

    print(f"Quantized versions of {base_model_id}:")
    for model in models:
        # Filter for common quantization tags or suffixes
        if any(q in model.id.upper() for q in ["GGUF", "AWQ", "GPTQ", "EXL2", "ONNX"]):
            print(model)
            print(f"- {model.id} ({model.downloads} downloads)")

    # for a given quantization, get the model size
    for model in models:
        if any(q in model.id.upper() for q in ["GGUF", "AWQ", "GPTQ", "EXL2", "ONNX"]):
            print(f"- {model.id} ({model.downloads} downloads)")


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

    print(f"System Resources:")
    print(f"  RAM: {format_to_human_readable(ram)}")
    print(f"  VRAM: {format_to_human_readable(vram)} (Total across {gpu_count} GPUs)")
    print(f"  Target Context: {args.context}")
    print(f"  Model Size: {get_model_size(model_info)/1e9:.2f}B parameters")
    print("-" * 32)

if __name__ == "__main__":
    main()