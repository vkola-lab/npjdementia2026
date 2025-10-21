"""
Discovery Agent vLLM Client

This script demonstrates how to use the vLLM-based ModelClient
for the Discovery Agent system with multi-GPU support.
"""

import os
import sys
import time
import logging
import torch
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for vLLM optimization
os.environ["HF_HOME"] = "/projectnb/vkolagrp/ketanss/.cache"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"


class ModelClient:
    """vLLM-based model client with multi-GPU support and same interface as HuggingFace client."""
    
    def __init__(self, config=None):
        """
        Initialize vLLM model client.
        
        Args:
            config: Configuration object with model_name, device, max_new_tokens, max_model_len attributes
        """
        # Use provided config or fall back to defaults
        if config is None:
            self.model_name = "Qwen/Qwen2.5-32B-Instruct"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.max_new_tokens = 8192
            self.max_model_len = 32768
        else:
            self.model_name = config.model_name
            self.device = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_new_tokens = config.max_new_tokens
            self.max_model_len = getattr(config, 'max_model_len', 32768)  # Default to 32768 if not specified
        
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
                max_model_len=self.max_model_len,  # Configurable context length
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for vLLM optimization
os.environ["HF_HOME"] = "/projectnb/vkolagrp/ketanss/.cache"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"


class DiscoveryAgentConfig:
    """Simple config class for Discovery Agent vLLM client."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-32B-Instruct", device="auto", max_tokens=8192, max_model_len=32768):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_tokens
        self.max_model_len = max_model_len


def test_model_client():
    """Test the vLLM ModelClient with various scenarios."""
    print("=" * 60)
    print("DISCOVERY AGENT vLLM CLIENT TEST")
    print("=" * 60)
    
    # Validate GPU setup
    print("\n🔍 Validating GPU setup...")
    gpu_valid = validate_gpu_setup()
    
    if not gpu_valid:
        print("⚠️  GPU setup validation failed, but continuing with CPU...")
    
    # Initialize configuration
    config = DiscoveryAgentConfig(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        device="auto",
        max_tokens=2048
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Max tokens: {config.max_new_tokens}")
    
    try:
        # Initialize vLLM ModelClient
        print(f"\n🚀 Initializing vLLM ModelClient...")
        start_time = time.time()
        client = ModelClient(config)
        init_time = time.time() - start_time
        print(f"✅ ModelClient initialized in {init_time:.2f} seconds")
        
        # Test 1: Simple conversation
        print(f"\n📝 Test 1: Simple conversation")
        messages = [
            {"role": "user", "content": "Hello! Can you help me with a research question about Alzheimer's disease?"}
        ]
        
        start_time = time.time()
        response = client.generate(messages)
        generation_time = time.time() - start_time
        
        print(f"Response time: {generation_time:.2f} seconds")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Test 2: Multi-turn conversation
        print(f"\n📝 Test 2: Multi-turn conversation")
        messages = [
            {"role": "user", "content": "I'm studying the relationship between education and cognitive decline in Alzheimer's patients."},
            {"role": "assistant", "content": "That's an interesting research area. Education is often considered a protective factor against cognitive decline."},
            {"role": "user", "content": "What specific hypotheses could I test with this relationship?"}
        ]
        
        start_time = time.time()
        response = client.generate(messages)
        generation_time = time.time() - start_time
        
        print(f"Response time: {generation_time:.2f} seconds")
        print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Test 3: Hypothesis generation (Discovery Agent style)
        print(f"\n📝 Test 3: Hypothesis generation")
        messages = [
            {"role": "system", "content": "You are an expert hypothesis generator for Alzheimer's Disease and Related Dementias (ADRD) research."},
            {"role": "user", "content": "Generate 3 novel hypotheses about the relationship between physical activity and cognitive function in elderly populations."}
        ]
        
        start_time = time.time()
        response = client.generate(messages)
        generation_time = time.time() - start_time
        
        print(f"Response time: {generation_time:.2f} seconds")
        print(f"Response: {response[:300]}{'...' if len(response) > 300 else ''}")
        
        # Performance summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"  - GPU setup: {'✅ Valid' if gpu_valid else '⚠️ Issues detected'}")
        print(f"  - Available GPUs: {get_available_gpus()}")
        print(f"  - Tensor parallel size: {get_tensor_parallel_size()}")
        print(f"  - Model initialization: {init_time:.2f}s")
        print(f"  - Average generation time: {generation_time:.2f}s")
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def benchmark_performance():
    """Benchmark the vLLM client performance."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    config = DiscoveryAgentConfig(max_tokens=1024)
    
    try:
        client = ModelClient(config)
        
        # Benchmark with different prompt lengths
        test_cases = [
            ("Short prompt", "What is Alzheimer's disease?"),
            ("Medium prompt", "Explain the relationship between APOE genotype and Alzheimer's disease risk, including the mechanisms involved."),
            ("Long prompt", "Generate a comprehensive research hypothesis about the interaction between cardiovascular health, education level, and genetic factors in predicting cognitive decline in elderly populations. Consider both protective and risk factors.")
        ]
        
        results = []
        
        for test_name, prompt in test_cases:
            print(f"\n🧪 Testing: {test_name}")
            messages = [{"role": "user", "content": prompt}]
            
            # Run multiple iterations for average
            times = []
            for i in range(3):
                start_time = time.time()
                response = client.generate(messages)
                generation_time = time.time() - start_time
                times.append(generation_time)
            
            avg_time = sum(times) / len(times)
            token_count = len(response.split())
            tokens_per_sec = token_count / avg_time if avg_time > 0 else 0
            
            results.append({
                'test': test_name,
                'avg_time': avg_time,
                'tokens': token_count,
                'tokens_per_sec': tokens_per_sec
            })
            
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Tokens generated: {token_count}")
            print(f"  Tokens/second: {tokens_per_sec:.2f}")
        
        # Summary
        print(f"\n📊 BENCHMARK SUMMARY:")
        for result in results:
            print(f"  {result['test']}: {result['tokens_per_sec']:.2f} tokens/sec")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run Discovery Agent vLLM client tests."""
    print("Starting Discovery Agent vLLM Client Tests...")
    
    # Run basic tests
    success = test_model_client()
    
    if success:
        # Run performance benchmark
        benchmark_performance()
    
    print(f"\n🎯 Discovery Agent vLLM Client testing completed!")


if __name__ == "__main__":
    main()
