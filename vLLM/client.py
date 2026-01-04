"""
vLLM Multi-GPU Inference Client

This script provides vLLM inference with flexible multi-GPU support using default settings.
- Automatically detects and uses any number of available GPUs
- Uses tensor parallelism to split large models across all available GPUs
- Real-time GPU usage monitoring for all GPUs
- Falls back gracefully if GPU setup fails
- Minimal optimizations: Flash Attention + high memory utilization
- Comprehensive GPU utilization reporting
"""

import os
import time
import torch
import gc
import subprocess
import threading

# Set HuggingFace cache directory (optional - defaults to ~/.cache/huggingface)
# Uncomment and update the path below if you want to use a custom cache location
# os.environ["HF_HOME"] = "/path/to/your/huggingface/cache"
# Essential vLLM optimizations for speed and memory
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"  # Enable Flash Attention for speed
from vllm import LLM, SamplingParams

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT = "Write an essay on the topic of AI and its impact on society in 2000 words."
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# GPU Configuration
def get_available_gpus():
    """Get the number of available GPUs"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def get_tensor_parallel_size():
    """Determine optimal tensor parallel size based on available GPUs"""
    available_gpus = get_available_gpus()
    if available_gpus >= 1:
        return available_gpus  # Use all available GPUs for tensor parallelism
    else:
        return 1  # Fallback to CPU (though not recommended)

def validate_gpu_setup():
    """Validate GPU setup and provide recommendations"""
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

def get_gpu_usage():
    """Get GPU usage information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_info.append({
                            'index': int(parts[0]),
                            'utilization': int(parts[1]),
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'temperature': int(parts[4])
                        })
            return gpu_info
    except Exception as e:
        print(f"Warning: Could not get GPU usage info: {e}")
    return []

def print_gpu_usage():
    """Print current GPU usage for all available GPUs"""
    gpu_info = get_gpu_usage()
    if gpu_info:
        print("\n🖥️  GPU USAGE:")
        for gpu in gpu_info:
            memory_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
            print(f"  GPU {gpu['index']}: {gpu['utilization']:3d}% util | "
                  f"{gpu['memory_used']:5d}MB/{gpu['memory_total']:5d}MB ({memory_percent:5.1f}%) | "
                  f"{gpu['temperature']:2d}°C")
    else:
        print("\n🖥️  GPU USAGE: Unable to retrieve GPU information")

def monitor_gpu_usage(duration=10, interval=1):
    """Monitor GPU usage for a specified duration"""
    print(f"\n📊 Monitoring GPU usage for {duration} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        print_gpu_usage()
        time.sleep(interval)
    
    print("📊 GPU monitoring completed")

def vllm_inference():
    """Run inference using vLLM with optimizations"""
    print("=" * 50)
    print("vLLM INFERENCE (OPTIMIZED)")
    print("=" * 50)
    
    # Check GPU availability and configure tensor parallelism
    available_gpus = get_available_gpus()
    tensor_parallel_size = get_tensor_parallel_size()
    
    print(f"Available GPUs: {available_gpus}")
    print(f"Using tensor parallel size: {tensor_parallel_size}")
    
    if tensor_parallel_size > 1:
        print(f"🚀 Multi-GPU inference enabled: {tensor_parallel_size} GPUs")
        print("   Model will be split across GPUs using tensor parallelism")
    else:
        print("📱 Single GPU inference")
    
    # Initialize vLLM with default settings
    start_time = time.time()
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        max_tokens=MAX_TOKENS,
    )
    
    try:
        # Use default vLLM configuration with minimal essential optimizations
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=tensor_parallel_size,  # Multi-GPU support
            gpu_memory_utilization=0.9,  # Use most of GPU memory
            max_model_len=32768,  # Reasonable context length
        )
        print("✅ vLLM initialized with default settings + multi-GPU support")
    except Exception as e:
        print(f"⚠️  Multi-GPU setup failed: {e}")
        print("🔄 Falling back to single GPU...")
        # Fallback to single GPU with default settings
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,  # Single GPU
            gpu_memory_utilization=0.9,  # Use most of GPU memory
            max_model_len=32768,  # Reasonable context length
        )
        print("✅ vLLM initialized with single GPU")
    init_time = time.time() - start_time
    print(f"vLLM initialization time: {init_time:.2f} seconds")
    
    # Show GPU usage after initialization
    print_gpu_usage()
    
    # Run inference with single prompt
    start_time = time.time()
    
    # Start GPU monitoring in a separate thread during inference
    monitoring_thread = None
    if tensor_parallel_size > 1:
        print(f"\n🚀 Starting inference with {tensor_parallel_size}-GPU tensor parallelism...")
        # Monitor GPU usage during inference
        monitoring_thread = threading.Thread(target=monitor_gpu_usage, args=(30, 2))  # Monitor for 30 seconds, every 2 seconds
        monitoring_thread.daemon = True
        monitoring_thread.start()
    else:
        print(f"\n🚀 Starting inference with single GPU...")
    
    outputs = llm.generate(PROMPT, sampling_params)
    inference_time = time.time() - start_time
    
    # Show final GPU usage
    print_gpu_usage()
    
    # Calculate actual tokens generated
    total_tokens = len(outputs[0].outputs[0].text.split())
    actual_tokens_per_sec = total_tokens / inference_time
    
    print(f"vLLM inference time: {inference_time:.2f} seconds")
    print(f"vLLM total time (init + inference): {init_time + inference_time:.2f} seconds")
    print(f"vLLM tokens per second: {actual_tokens_per_sec:.2f}")
    print(f"vLLM total tokens generated: {total_tokens}")
    print("\nvLLM Output:")
    print(outputs[0].outputs[0].text[:500] + "..." if len(outputs[0].outputs[0].text) > 500 else outputs[0].outputs[0].text)
    
    return inference_time, init_time, total_tokens


def main():
    """Main function to run vLLM multi-GPU inference"""
    print("Starting vLLM Multi-GPU Inference...")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT[:100]}...")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    
    # Validate GPU setup
    gpu_setup_valid = validate_gpu_setup()
    
    # Display GPU information
    available_gpus = get_available_gpus()
    tensor_parallel_size = get_tensor_parallel_size()
    print(f"\n🖥️  GPU CONFIGURATION:")
    print(f"  - Available GPUs: {available_gpus}")
    print(f"  - Tensor parallel size: {tensor_parallel_size}")
    if tensor_parallel_size > 1:
        print(f"  - Multi-GPU mode: Model will be split across {tensor_parallel_size} GPUs")
    else:
        print(f"  - Single GPU mode")
    
    print("\n🚀 vLLM CONFIGURATION:")
    print("  - Default vLLM settings")
    print("  - Flash Attention for speed")
    print("  - High GPU memory utilization (90%)")
    if tensor_parallel_size > 1:
        print(f"  - Multi-GPU tensor parallelism ({tensor_parallel_size} GPUs)")
        print("  - Real-time GPU usage monitoring")
    else:
        print("  - Single GPU mode")
    
    # Show initial GPU usage
    print_gpu_usage()
    
    try:
        # Run vLLM inference
        vllm_inf_time, vllm_init_time, vllm_tokens = vllm_inference()
        
        # Performance summary
        print("\n" + "=" * 50)
        print("vLLM PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Initialization time: {vllm_init_time:.2f}s")
        print(f"Inference time: {vllm_inf_time:.2f}s")
        print(f"Total time: {vllm_init_time + vllm_inf_time:.2f}s")
        print(f"Tokens generated: {vllm_tokens}")
        print(f"Tokens per second: {vllm_tokens/vllm_inf_time:.2f}")
        
        if tensor_parallel_size > 1:
            print(f"\n🎯 Multi-GPU Performance:")
            print(f"  - Using {tensor_parallel_size} GPUs with tensor parallelism")
            print(f"  - Model split across {tensor_parallel_size} GPUs for optimal memory usage")
            print(f"  - Enhanced throughput with parallel processing")
        else:
            print(f"\n🎯 Single GPU Performance:")
            print(f"  - Using 1 GPU for inference")
            print(f"  - Standard single GPU processing")
        
        # Show final GPU usage
        print_gpu_usage()
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
