#!/usr/bin/env python3
"""
Test script for the updated vLLM client with timeout and fallback mechanisms.
"""

import sys
import os
import time
import logging

# Add the discovery_agent_pkg to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'discovery_agent_pkg'))

from discovery_agent_pkg.agents.vllm_client import ModelClient
from discovery_agent_pkg.core.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vllm_client():
    """Test the vLLM client with timeout and fallback mechanisms."""
    print("🧪 Testing updated vLLM client...")
    
    try:
        # Load configuration
        config = Config()
        print(f"📋 Using model: {config.model_name}")
        print(f"📋 Device: {config.device}")
        print(f"📋 Client type: {config.model_client_type}")
        
        # Initialize client
        print("🚀 Initializing vLLM client...")
        start_time = time.time()
        client = ModelClient(config)
        init_time = time.time() - start_time
        print(f"✅ Client initialized in {init_time:.2f} seconds")
        
        # Test generation with a simple prompt
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with a short greeting."}
        ]
        
        print("🤖 Testing generation...")
        start_time = time.time()
        response = client.generate(test_messages, timeout=60)  # 60 second timeout
        gen_time = time.time() - start_time
        
        print(f"✅ Generation completed in {gen_time:.2f} seconds")
        print(f"📝 Response: {response[:100]}...")  # Show first 100 characters
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_vllm_client()
    if success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Tests failed!")
        sys.exit(1)
