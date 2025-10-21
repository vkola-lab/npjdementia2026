"""
vLLM Model Client for Discovery Agent

This module provides a ModelClient class that uses vLLM for inference,
maintaining the same interface as the original HuggingFace-based client.
"""

import os
import time
import torch
import logging
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# Environment variables will be set from config in ModelClient.__init__


class ModelClient:
    """vLLM-based model client with multi-GPU support and same interface as HuggingFace client."""
    
    def __init__(self, config=None):
        """
        Initialize vLLM model client.
        
        Args:
            config: Configuration object with model_name, device, max_new_tokens, and vllm_env attributes
        """
        # Use provided config or fall back to defaults
        if config is None:
            self.model_name = "Qwen/Qwen2.5-32B-Instruct"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.max_new_tokens = 8192
            vllm_env = {"VLLM_USE_TRITON_FLASH_ATTN": "1"}
        else:
            self.model_name = config.model_name
            self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_new_tokens = config.max_new_tokens
            vllm_env = getattr(config, 'vllm_env', {"VLLM_USE_TRITON_FLASH_ATTN": "1"})
        
        # Set vLLM environment variables from config
        for key, value in vllm_env.items():
            os.environ[key] = str(value)
            logger.info(f"Set vLLM environment variable: {key}={value}")
        
        print(f"Loading vLLM model: {self.model_name}")
        logger.info(f"Initializing vLLM client with device: {self.device}")
        
        try:
            # Determine tensor parallel size from available GPUs
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            tensor_parallel_size = available_gpus if available_gpus > 0 else 1
            
            if tensor_parallel_size > 1:
                print(f"🚀 Using tensor parallelism across {tensor_parallel_size} GPUs")
            elif available_gpus == 1:
                print("📱 Using single GPU")
            else:
                print("⚠️ No GPU detected, running on CPU (slow)")
            
            # Initialize vLLM engine with optimizations
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9,  # Use most of GPU memory
                max_model_len=60000,  # Reasonable context length
                trust_remote_code=True,
            )
            
            print("✅ vLLM model loaded successfully")
            logger.info("vLLM client initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize vLLM client: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """
        Generate text response from messages with retry mechanism.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated text response
        """
        def format_messages_as_prompt(msgs: List[Dict[str, Any]]) -> str:
            """Format messages as a simple role-tagged prompt."""
            transcript = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                transcript.append(f"<{role}>\n{content}\n</{role}>")
            return "\n\n".join(transcript)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=self.max_new_tokens,
        )
        
        prompt = format_messages_as_prompt(messages)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response with vLLM (attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                
                # Generate using vLLM
                outputs = self.llm.generate(prompt, sampling_params)
                
                generation_time = time.time() - start_time
                logger.info(f"Generation completed in {generation_time:.2f} seconds")
                
                # Extract generated text
                if not outputs or not outputs[0].outputs:
                    raise ValueError("Empty outputs from vLLM")
                
                text = outputs[0].outputs[0].text
                if not text or not text.strip():
                    raise ValueError("Generated empty response")
                
                return text.strip()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    torch.cuda.empty_cache()
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError("CUDA out of memory after all retries") from e
                    
            except Exception as e:
                logger.error(f"Generation error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise RuntimeError(f"Generation failed after {max_retries} attempts: {e}") from e
        
        raise RuntimeError("All generation attempts failed")


def get_available_gpus():
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_tensor_parallel_size():
    """Determine optimal tensor parallel size based on available GPUs."""
    available_gpus = get_available_gpus()
    if available_gpus >= 1:
        return available_gpus  # Use all available GPUs for tensor parallelism
    else:
        return 1  # Fallback to CPU (though not recommended)


def validate_gpu_setup():
    """Validate GPU setup and provide recommendations."""
    available_gpus = get_available_gpus()
    tensor_parallel_size = get_tensor_parallel_size()
    
    print(f"\n🔍 GPU SETUP VALIDATION:")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        print("  ⚠️  WARNING: No GPUs detected! Performance will be very poor.")
        return False
    elif available_gpus == 1:
        print("  ℹ️  Single GPU detected. Using single GPU mode.")
        return True
    else:
        print(f"  ✅ {available_gpus} GPUs detected. Using {tensor_parallel_size}-GPU tensor parallelism.")
        print(f"  📊 Model will be split across {tensor_parallel_size} GPUs for better performance.")
        return True
    
    return True


def main():
    """Test function to demonstrate vLLM ModelClient usage."""
    print("Testing vLLM ModelClient...")
    
    # Validate GPU setup
    validate_gpu_setup()
    
    try:
        # Initialize client
        client = ModelClient()
        
        # Test generation
        test_messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        print("\n🚀 Testing generation...")
        response = client.generate(test_messages)
        print(f"Response: {response}")
        
        print("\n✅ vLLM ModelClient test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
