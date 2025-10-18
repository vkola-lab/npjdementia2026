import os
import torch
from typing import List, Dict, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

# Set environment variables
os.environ["HF_HOME"] = "/projectnb/vkolagrp/ketanss/.cache"


def check_gpu_availability():
    """Check available GPUs and their memory."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return gpu_count
    else:
        print("⚠️  No CUDA GPUs available")
        return 0

class Qwen30BClient:
    """
    Client for loading and using Qwen model with direct model loading.
    Automatically detects GPU availability and falls back to CPU if needed.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_tokens: int = 16384,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 gpu_memory_per_device: str = "20GB"):
        """
        Initialize the Qwen 30B client.
        
        Args:
            model_name: HuggingFace model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            gpu_memory_per_device: Memory allocation per GPU (e.g., "20GB")
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_per_device = gpu_memory_per_device
        
        self._init_direct_model()
    
    def _init_direct_model(self):
        """Initialize direct model loading."""
        try:
            print(f"Loading {self.model_name}...")
            
            # Check GPU availability
            gpu_count = check_gpu_availability()
            use_cuda = gpu_count > 0
            
            if not use_cuda:
                print("⚠️  No CUDA GPUs available. Loading model on CPU.")
            else:
                print(f"✓ Found {gpu_count} GPU(s). Loading model on GPU.")
            
            # Load tokenizer with caching
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=os.environ.get("HF_HOME", "/projectnb/vkolagrp/ketanss/.cache")
            )
            
            # Load model with appropriate device configuration
            if use_cuda:
                # Load model with GPU optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
                    device_map="auto",  # Let transformers automatically split across available GPUs
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=os.environ.get("HF_HOME", "/projectnb/vkolagrp/ketanss/.cache")
                )
                print("✓ Model loaded successfully on GPU(s)")
            else:
                # Load model for CPU usage
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=os.environ.get("HF_HOME", "/projectnb/vkolagrp/ketanss/.cache")
                )
                print("✓ Model loaded successfully on CPU")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate(self, 
                 prompt: str, 
                 messages: Optional[List[Dict[str, str]]] = None,
                 **kwargs) -> str:
        """
        Generate text using the Qwen model.
        
        Args:
            prompt: Input prompt string
            messages: List of message dictionaries (for chat format)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        return self._generate_direct(prompt, messages, **kwargs)
    
    def _generate_direct(self, 
                        prompt: str, 
                        messages: Optional[List[Dict[str, str]]] = None,
                        **kwargs) -> str:
        """Generate using direct model loading."""
        try:
            print("🔄 Preparing input...")
            if messages:
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = prompt
            
            print(f"📝 Input text length: {len(text)} characters")
            
            # Tokenize input
            print("🔤 Tokenizing input...")
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_length = model_inputs.input_ids.shape[1]
            print(f"📊 Input tokens: {input_length}")
            
            # Set generation parameters
            max_new_tokens = min(kwargs.get('max_new_tokens', self.max_tokens), 512)  # Limit to 512 for testing
            print(f"🎯 Max new tokens: {max_new_tokens}")
            
            # Generate
            print("🚀 Starting generation...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=kwargs.get('temperature', self.temperature),
                    top_p=kwargs.get('top_p', self.top_p),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            print("✅ Generation completed!")
            
            # Decode output
            print("🔍 Decoding output...")
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            print(f"📄 Generated {len(content)} characters")
            return content
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise RuntimeError(f"Direct generation failed: {e}")
    
    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Chat interface for the model.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": message})
        
        return self.generate("", messages=messages)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to generate for prompt: {e}")
                results.append("")
        
        return results
    
    def save_model_to_cache(self, save_directory: str = None):
        """
        Save the loaded model to a local directory for faster future loading.
        
        Args:
            save_directory: Directory to save the model (defaults to cache directory)
        """
        if save_directory is None:
            save_directory = os.path.join(os.environ.get("HF_HOME", "/projectnb/vkolagrp/ketanss/.cache"), "qwen-30b-local")
        
        try:
            print(f"Saving model to {save_directory}...")
            self.model.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)
            print(f"✓ Model saved successfully to {save_directory}")
            return save_directory
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
    
    def load_from_local_cache(self, local_path: str):
        """
        Load model from a local cache directory.
        
        Args:
            local_path: Path to the local model directory
        """
        try:
            print(f"Loading model from local cache: {local_path}...")
            
            # Load tokenizer from local cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                trust_remote_code=True
            )
            
            # Load model from local cache
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: self.gpu_memory_per_device, 1: self.gpu_memory_per_device}
            )
            
            print("✓ Model loaded successfully from local cache")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from local cache: {e}")

def main():
    """Example usage of the Qwen30BClient."""
    
    print("=== Qwen Model Loading Example ===")
    
    # Check if local cache exists
    local_cache_path = os.path.join(os.environ.get("HF_HOME", "/projectnb/vkolagrp/ketanss/.cache"), "qwen-30b-local")
    
    try:
        if os.path.exists(local_cache_path):
            print("Found local cache, loading from cache...")
            client = Qwen30BClient(
                model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
                max_tokens=128,  # Reduced for testing
                gpu_memory_per_device="20GB"
            )
            client.load_from_local_cache(local_cache_path)
        else:
            print("No local cache found, downloading and caching model...")
            client = Qwen30BClient(
                model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
                max_tokens=128,  # Reduced for testing
                gpu_memory_per_device="20GB"  # Split across first 2 GPUs with 20GB each
            )
            
            # Save model to local cache for future use
            client.save_model_to_cache(local_cache_path)
        
        print("\n" + "="*50)
        print("TESTING SIMPLE GENERATION")
        print("="*50)
        
        # Simple generation with a very short prompt
        response = client.generate("Hello, how are you?", max_new_tokens=50)
        print("Response:", response)
        
        print("\n" + "="*50)
        print("TESTING CHAT INTERFACE")
        print("="*50)
        
        # Chat interface with a simple question
        chat_response = client.chat(
            "What is 2+2?",
            system_prompt="You are a helpful math tutor."
        )
        print("Chat Response:", chat_response)
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()