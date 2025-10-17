#!/bin/bash -l
# ------------------------------------------------------------------------------
# SCC directives
# ------------------------------------------------------------------------------
#$ -P vkolagrp
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l h_rt=48:00:00
#$ -N discoAD_with_planning
#$ -j y
#$ -m ea
#$ -o /projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/DiscoADRD_v2/analysis/qsub_runs

# ------------------------------------------------------------------------------
# Configuration - Edit these values to customize your setup
# ------------------------------------------------------------------------------

# User input configuration - Edit these values for your specific analysis
HYPOTHESIS_MODE="csv_batch"  # Options: "generate", "direct", "direct hypothesis", or "csv_batch"

# ------------------------------------------------------------------------------
# System Setup
# ------------------------------------------------------------------------------

# Change to the DiscoADRD_v2 directory
cd /projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/DiscoADRD_v2

# nvidia-smi
module load miniconda
module load cuda
conda activate llms

# ------------------------------------------------------------------------------
# Run Discovery Agent with Configuration
# ------------------------------------------------------------------------------

echo "🚀 Starting DiscoADRD_v2 Discovery Agent..."
echo "🔧 Configuration:"
echo "   Hypothesis mode: $HYPOTHESIS_MODE"
echo "   (All other settings loaded from config.yml)"
echo ""

echo "🎯 Running Discovery Agent..."
echo "=========================================="

# Run the discovery agent
python discover.py \
    --hypothesis-mode "$HYPOTHESIS_MODE"

echo "✅ Completed run"
echo "🎉 All runs completed!"
    