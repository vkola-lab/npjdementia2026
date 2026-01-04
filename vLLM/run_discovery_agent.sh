#!/bin/bash

# Discovery Agent vLLM Runner Script
# This script runs the Discovery Agent with vLLM backend

echo "🚀 Starting Discovery Agent with vLLM backend..."

# Set environment variables
# Update HF_HOME to your HuggingFace cache directory
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # Use environment variable or default to $HOME/.cache/huggingface
export VLLM_USE_TRITON_FLASH_ATTN="1"

# Change to the vLLM directory
cd "$(dirname "$0")"

# Run the Discovery Agent vLLM client
echo "📋 Running Discovery Agent vLLM client tests..."
module load miniconda
module load cuda
conda activate llms
python discovery_agent_client.py

echo "✅ Discovery Agent vLLM testing completed!"
