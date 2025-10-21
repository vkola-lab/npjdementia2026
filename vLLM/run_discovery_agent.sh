#!/bin/bash

# Discovery Agent vLLM Runner Script
# This script runs the Discovery Agent with vLLM backend

echo "🚀 Starting Discovery Agent with vLLM backend..."

# Set environment variables
export HF_HOME="/projectnb/vkolagrp/ketanss/.cache"
export VLLM_USE_TRITON_FLASH_ATTN="1"

# Change to the vLLM directory
cd "$(dirname "$0")"

# Run the Discovery Agent vLLM client
echo "📋 Running Discovery Agent vLLM client tests..."
cd /projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/vLLM
module load miniconda
module load cuda
conda activate llms
python discovery_agent_client.py

echo "✅ Discovery Agent vLLM testing completed!"
