#!/usr/bin/env python3
"""
Test script for the updated vLLM client with native chat() method.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discovery_agent_client import ModelClient

def test_chat_functionality():
    """Test the updated vLLM client with native chat() method."""
    print("🧪 Testing vLLM client with native chat() method...")
    
    try:
        # Initialize client
        client = ModelClient()
        
        # Test with system and user messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum entanglement simply."}
        ]
        
        print("📝 Testing with messages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")
        
        print("\n🚀 Generating response...")
        response = client.generate(messages)
        
        print(f"\n✅ Response received:")
        print(f"🤖 Assistant: {response}")
        
        # Test with conversation history
        print("\n" + "="*50)
        print("🔄 Testing conversation history...")
        
        conversation_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is the population of that city?"}
        ]
        
        print("📝 Testing with conversation:")
        for msg in conversation_messages:
            print(f"  {msg['role']}: {msg['content']}")
        
        print("\n🚀 Generating response...")
        response2 = client.generate(conversation_messages)
        
        print(f"\n✅ Response received:")
        print(f"🤖 Assistant: {response2}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chat_functionality()
    sys.exit(0 if success else 1)
