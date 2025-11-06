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
#$ -N wo_plan_45
#$ -j y
#$ -m ea
#$ -o /projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/DiscoADRD_v2/analysis/qsub_runs

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
echo "   All settings loaded from config.yml"
echo ""

echo "🎯 Running Discovery Agent..."
echo "=========================================="

# Run the discovery agent (all configuration from config.yml)
python discover.py 

echo "✅ Completed run"
echo "🎉 All runs completed!"
    