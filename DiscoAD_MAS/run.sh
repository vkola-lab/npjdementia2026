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
#$ -N discoAD
#$ -j y
#$ -m ea
#$ -o /projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/DiscoAD_MAS/analysis/qsub_runs

# ------------------------------------------------------------------------------
# Configuration - Edit these values to customize your setup
# ------------------------------------------------------------------------------

# User input configuration - Edit these values for your specific analysis
HYPOTHESIS_MODE="csv_batch"  # Options: "generate", "direct", "direct hypothesis", or "csv_batch"
HYPOTHESIS="APOE ε4 carriage increases risk for Alzheimer's disease across ethnicities."
# HYPOTHESIS="Greater years of education are linked to lower dementia incidence (cognitive reserve)."
# HYPOTHESIS="Greater years of education are linked to higher dementia incidence"
CSV_FILE="/projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/DiscoAD_MAS/batch.csv"  # Path to CSV file containing hypotheses (only needed for csv_batch mode)
START_INDEX=25  # Start processing CSV from this index (1-based, default: 1)
DATASET_FOLDER="/projectnb/vkolagrp/ketanss/adrd_agent_custom/adrd_agent/datasets/NACC"

# Output verbosity control
VERBOSE=true  # Set to "true" for detailed output, "false" for quiet mode

# ------------------------------------------------------------------------------
# System Setup
# ------------------------------------------------------------------------------

# Change to the DiscoAD_MAS directory
cd /projectnb/vkolagrp/ketanss/adrd_agent_custom/DiscoADRD/DiscoAD_MAS

# nvidia-smi
module load miniconda
module load cuda
conda activate llms

# ------------------------------------------------------------------------------
# Run Discovery Agent with Configuration
# ------------------------------------------------------------------------------

echo "🚀 Starting DiscoAD_MAS Discovery Agent..."
echo "🔧 Configuration:"
echo "   Hypothesis mode: $HYPOTHESIS_MODE"
if [ "$HYPOTHESIS_MODE" = "direct hypothesis" ]; then
    echo "   Direct hypothesis: $HYPOTHESIS"
elif [ "$HYPOTHESIS_MODE" = "csv_batch" ]; then
    echo "   CSV file: $CSV_FILE"
    if [ "$START_INDEX" != "1" ]; then
        echo "   Start index: $START_INDEX"
    fi
fi
echo "   Dataset folder: $DATASET_FOLDER"
echo "   Verbose output: $VERBOSE"
echo "   (Other settings loaded from config.yml)"
echo ""

# Check if using direct hypothesis mode
if [ "$HYPOTHESIS_MODE" = "direct hypothesis" ]; then
    echo "🎯 Running with direct hypothesis (single call)..."
    echo "=========================================="
    
    # Run the discovery agent with direct hypothesis (single call)
    if [ "$VERBOSE" = "true" ]; then
        python discover.py \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --hypothesis "$HYPOTHESIS" \
            --dataset-folder "$DATASET_FOLDER" \
            --verbose
    else
        python discover.py \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --hypothesis "$HYPOTHESIS" \
            --dataset-folder "$DATASET_FOLDER"
    fi
    
    echo "✅ Completed direct hypothesis run"
elif [ "$HYPOTHESIS_MODE" = "csv_batch" ]; then
    echo "📊 Running with CSV batch mode..."
    echo "=========================================="
    
    # Run the discovery agent with CSV batch mode
    if [ "$VERBOSE" = "true" ]; then
        python discover.py \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --csv-file "$CSV_FILE" \
            --start-index "$START_INDEX" \
            --dataset-folder "$DATASET_FOLDER" \
            --verbose
    else
        python discover.py \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --csv-file "$CSV_FILE" \
            --start-index "$START_INDEX" \
            --dataset-folder "$DATASET_FOLDER"
    fi
    
    echo "✅ Completed CSV batch run"
else
    # Run for generate/direct modes
    echo "🔄 Running with hypothesis generation mode..."
    echo "=========================================="
    
    # Run the discovery agent
    if [ "$VERBOSE" = "true" ]; then
        python discover.py \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --dataset-folder "$DATASET_FOLDER" \
            --verbose
    else
        python discover.py \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --dataset-folder "$DATASET_FOLDER"
    fi
    
    echo "✅ Completed run"
fi

echo "🎉 All runs completed!"
    