"""Model estimation utilities for LLM quantization."""

from functools import lru_cache
from huggingface_hub import hf_hub_download
import json

# Model architecture constants
DEFAULT_HEAD_DIM = 64
ESTIMATED_OVERHEAD = 1.2


class ModelEstimator:
    """Handles model information retrieval and size estimation."""
    
    def __init__(self, model_id: str):
        """Initialize the estimator with a model ID.
        
        Args:
            model_id: Hugging Face model ID
        """
        self.model_id = model_id
        self._config = None
    
    @property
    def config(self) -> dict:
        """Get model configuration (cached).
        
        Returns:
            Model configuration dictionary
        """
        if self._config is None:
            self._config = self._fetch_config()
        return self._config
    
    def _fetch_config(self) -> dict:
        """Fetch model configuration from Hugging Face.
        
        Returns:
            Model configuration dictionary
        """
        config_path = hf_hub_download(repo_id=self.model_id, filename="config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    
    def get_parameter_count(self) -> int:
        """Calculate total model parameters.
        
        Returns:
            Total number of parameters
        """
        config = self.config
        
        # 1. Embeddings (input + output)
        # Note: GPT-OSS usually does not tie word embeddings, so we count them twice
        embeddings = 2 * (config["vocab_size"] * config["hidden_size"])
        
        # 2. Attention Block (Per Layer)
        # For GQA: (num_heads + 2 * num_kv_heads) * (head_dim * hidden_size) + (hidden_size^2)
        head_dim = config.get("head_dim", DEFAULT_HEAD_DIM)
        attn_per_layer = (config["num_attention_heads"] + 2 * config["num_key_value_heads"]) * \
                        (head_dim * config["hidden_size"]) + (config["hidden_size"]**2)
        
        # 3. MoE Experts (Per Layer)
        # Each expert is a SwiGLU MLP: 3 matrices of (hidden_size * intermediate_size)
        expert_size = 3 * (config["hidden_size"] * config["intermediate_size"])
        all_experts_per_layer = config["num_local_experts"] * expert_size
        
        # 4. Sum it all up
        total_params = embeddings + config["num_hidden_layers"] * (attn_per_layer + all_experts_per_layer)
        return total_params
    
    def estimate_kv_cache_size(self, context: int, batch: int, kv_bits: int = 16) -> float:
        """Estimate KV cache memory requirements.
        
        Args:
            context: Context length (sequence length)
            batch: Batch size
            kv_bits: KV cache quantization bits (default 16)
            
        Returns:
            Estimated KV cache size in bytes
        """
        config = self.config
        num_attention_heads = config.get("num_attention_heads")
        num_kv_heads = config.get("num_key_value_heads", num_attention_heads)
        hidden_size = config.get("hidden_size")
        num_hidden_layers = config.get("num_hidden_layers")
        
        if num_attention_heads and hidden_size and num_hidden_layers:
            head_dim = hidden_size // num_attention_heads
            # Formula: (kv_bits / 8) bytes * 2 (K + V) * layers * kv_heads * head_dim * context * batch
            bytes_per_param = kv_bits / 8
            estimated_kv_cache = bytes_per_param * 2 * num_hidden_layers * num_kv_heads * head_dim * context * batch
        else:
            # Fallback if config is missing details
            bytes_per_param = kv_bits / 8
            hidden_size = config.get("hidden_size", 0)
            estimated_kv_cache = context * batch * hidden_size * bytes_per_param
        
        return estimated_kv_cache
    
    def estimate_quantized_size(self, quantization: str, quant_factor: float, 
                               context: int, batch: int, kv_bits: int = 16) -> float:
        """Estimate total memory requirements for a quantized model.
        
        Args:
            quantization: Quantization type (e.g., 'q4_k_m', 'fp16')
            quant_factor: Quantization factor (bytes per parameter)
            context: Context length
            batch: Batch size
            kv_bits: KV cache quantization bits
            
        Returns:
            Estimated total memory in bytes
        """
        model_size = self.get_parameter_count()
        kv_cache_size = self.estimate_kv_cache_size(context, batch, kv_bits)
        
        estimated_size = model_size * quant_factor * ESTIMATED_OVERHEAD + kv_cache_size
        return estimated_size
