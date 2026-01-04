#!/bin/bash -l
# ------------------------------------------------------------------------------
# SCC directives
# ------------------------------------------------------------------------------
#$ -P vkolagrp
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l h_rt=12:00:00
#$ -N wo_plan_42
#$ -j y
#$ -m ea
#$ -o analysis/qsub_runs  # Output directory for job logs (relative to working directory)

# ------------------------------------------------------------------------------
# System Setup
# ------------------------------------------------------------------------------

# Change to the DiscoADRD_v3 directory
# Update this path to match your installation directory
cd "$(dirname "$0")"  # Automatically change to script directory

# nvidia-smi
module load miniconda
module load cuda
conda activate llms

# ------------------------------------------------------------------------------
# Run Discovery Agent with Configuration
# ------------------------------------------------------------------------------

echo "🚀 Starting DiscoADRD_v3 Discovery Agent..."
echo "🔧 Configuration:"
echo "   All settings loaded from config.yml"
echo ""

echo "🎯 Running Discovery Agent..."
echo "=========================================="

# Run the discovery agent (all configuration from config.yml)
python discover.py 

echo "✅ Completed run"
echo "🎉 All runs completed!"
    