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
            self.max_model_len = 262144
            vllm_env = {"VLLM_USE_TRITON_FLASH_ATTN": "1"}
        else:
            self.model_name = config.model_name
            self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_new_tokens = config.max_new_tokens
            self.max_model_len = config.max_model_len
            self.seed = getattr(config, 'seed', 42)  # Get seed from config
            vllm_env = getattr(config, 'vllm_env', {"VLLM_USE_TRITON_FLASH_ATTN": "1"})
            self.sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                seed=self.seed,
            )
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
                max_model_len=self.max_model_len,  # Reasonable context length
                trust_remote_code=True,
            )
            
            print("✅ vLLM model loaded successfully")
            
            # Display model information
            self._display_model_info()
            
            logger.info("vLLM client initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize vLLM client: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _display_model_info(self):
        """Display model context length and other relevant information."""
        try:
            # Get model config from vLLM engine
            engine = self.llm.llm_engine
            model_config = engine.model_config
            
            # Try to get max context length from various sources
            max_length = None
            
            # Check vLLM's max_model_len setting
            if hasattr(engine, 'max_model_len'):
                max_length = engine.max_model_len
            
            # Check model config attributes
            if max_length is None and hasattr(model_config, 'max_model_len'):
                max_length = model_config.max_model_len
            elif max_length is None and hasattr(model_config, 'max_position_embeddings'):
                max_length = model_config.max_position_embeddings
            elif max_length is None and hasattr(model_config, 'n_positions'):
                max_length = model_config.n_positions
            elif max_length is None and hasattr(model_config, 'max_sequence_length'):
                max_length = model_config.max_sequence_length
            
            # Get tensor parallel info
            tensor_parallel_size = getattr(engine, 'tensor_parallel_size', 1)
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            print(f"📏 Model context length: {max_length if max_length else 'Unknown'}")
            print(f"🔧 Model name: {self.model_name}")
            print(f"💾 Device: {self.device}")
            print(f"🎯 Max new tokens: {self.max_new_tokens}")
            print(f"🚀 Tensor parallel size: {tensor_parallel_size}")
            print(f"🖥️ Available GPUs: {available_gpus}")
            
            # Display detailed GPU information
            self._display_gpu_info()
            
            # Show available config attributes for debugging
            if max_length is None:
                print("🔍 Available model config attributes:")
                for attr in dir(model_config):
                    if not attr.startswith('_') and 'length' in attr.lower():
                        try:
                            value = getattr(model_config, attr)
                            print(f"   - {attr}: {value}")
                        except:
                            pass
                            
        except Exception as e:
            print(f"⚠️ Could not determine model context length: {e}")
    
    def _display_gpu_info(self):
        """Display detailed GPU memory and utilization information."""
        try:
            if not torch.cuda.is_available():
                print("🔍 GPU Info: CUDA not available")
                return
            
            available_gpus = torch.cuda.device_count()
            print(f"\n🔍 GPU DETAILED INFORMATION:")
            print(f"  📊 Total GPUs: {available_gpus}")
            
            for i in range(available_gpus):
                # Get GPU properties
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                total_memory = props.total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                cached_memory = torch.cuda.memory_reserved(i)
                free_memory = total_memory - allocated_memory
                
                # Convert to GB
                total_gb = total_memory / (1024**3)
                allocated_gb = allocated_memory / (1024**3)
                cached_gb = cached_memory / (1024**3)
                free_gb = free_memory / (1024**3)
                
                # Calculate utilization percentages
                allocated_pct = (allocated_memory / total_memory) * 100
                cached_pct = (cached_memory / total_memory) * 100
                free_pct = (free_memory / total_memory) * 100
                
                print(f"  🎮 GPU {i} ({props.name}):")
                print(f"    💾 Total Memory: {total_gb:.2f} GB")
                print(f"    📈 Allocated: {allocated_gb:.2f} GB ({allocated_pct:.1f}%)")
                print(f"    🗂️  Cached: {cached_gb:.2f} GB ({cached_pct:.1f}%)")
                print(f"    🆓 Free: {free_gb:.2f} GB ({free_pct:.1f}%)")
                print(f"    🔧 Compute Capability: {props.major}.{props.minor}")
                print(f"    🧮 Multiprocessors: {props.multi_processor_count}")
                print(f"    ⚡ Max Threads per Block: {props.max_threads_per_block}")
                print(f"    📏 Max Threads per Multiprocessor: {props.max_threads_per_multiprocessor}")
                
                # Show current GPU utilization if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                    
                    print(f"    📊 GPU Utilization: {utilization.gpu}%")
                    print(f"    🧠 Memory Utilization: {utilization.memory}%")
                    print(f"    🌡️  Temperature: {temperature}°C")
                    print(f"    ⚡ Power Usage: {power_usage:.1f}W / {power_limit:.1f}W")
                    
                except ImportError:
                    print(f"    ℹ️  Install pynvml for detailed GPU monitoring")
                except Exception as e:
                    print(f"    ⚠️  Could not get GPU utilization: {e}")
                
                print()  # Empty line for readability
                
        except Exception as e:
            print(f"⚠️ Could not get GPU information: {e}")
    
    def generate(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """
        Generate text response from messages using vLLM's native chat() method.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated text response
        """

        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response with vLLM chat() (attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                
                # Use vLLM's native chat method
                outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
                
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
