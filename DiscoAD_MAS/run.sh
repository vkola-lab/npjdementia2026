#!/bin/bash -l
# ------------------------------------------------------------------------------
# SCC directives
# ------------------------------------------------------------------------------
#$ -P vkolagrp
#$ -t 1-1
#$ -pe omp 16
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l h_rt=48:00:00
#$ -N discoAD
#$ -j y
#$ -m ea
#$ -o /projectnb/vkolagrp/kelzokm/DiscoADRD/DiscoAD_MAS/qsub_runs

# ------------------------------------------------------------------------------
# Configuration - Edit these values to customize your setup
# ------------------------------------------------------------------------------

# Model configuration
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
DEVICE="auto"
MAX_TOKENS=16384

# Agent behavior
MAX_CONVERSATION_LENGTH=7
AUTO_APPROVE_AFTER=4
OUTER_LOOP_ITERATIONS=2

# User input configuration - Edit these values for your specific analysis
HYPOTHESIS_MODE="direct hypothesis"  # Options: "generate" or "direct hypothesis"
HYPOTHESIS="Higher age is associated with increased dementia incidence."  # Direct hypothesis when HYPOTHESIS_MODE is "direct hypothesis"
KEYWORDS_FILE="adrd_keywords.txt"  # Path to keywords file for sampling
# NUM_KEYWORDS will be set in the loop (2 and 3)
DATASET_FOLDER="/projectnb/vkolagrp/ketanss/adrd_agent_custom/adrd_agent/datasets/NACC"
NUM_SAMPLING_RUNS=10  # Number of keyword sampling iterations

# Output verbosity control
VERBOSE=true  # Set to "true" for detailed output, "false" for quiet mode

# ------------------------------------------------------------------------------
# System Setup
# ------------------------------------------------------------------------------

# Change to the DiscoAD_MAS directory
cd /projectnb/vkolagrp/kelzokm/DiscoADRD/DiscoAD_MAS

# nvidia-smi
module load miniconda
module load cuda
conda activate /projectnb/vkolagrp/ketanss/.conda/envs/llms

# ------------------------------------------------------------------------------
# Run Discovery Agent with Configuration
# ------------------------------------------------------------------------------

echo "🚀 Starting DiscoAD_MAS Discovery Agent..."
echo "🔧 Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Device: $DEVICE"
echo "   Max tokens: $MAX_TOKENS"
echo "   Hypothesis mode: $HYPOTHESIS_MODE"
if [ "$HYPOTHESIS_MODE" = "direct hypothesis" ]; then
    echo "   Direct hypothesis: $HYPOTHESIS"
fi
echo "   Keywords file: $KEYWORDS_FILE"
echo "   Number of keywords per run: 2 and 3 (looping)"
echo "   Number of sampling runs: $NUM_SAMPLING_RUNS"
echo "   Outer loop iterations: $OUTER_LOOP_ITERATIONS"
echo "   Verbose output: $VERBOSE"
if [ -n "$DATASET_FOLDER" ]; then
    echo "   Dataset folder: $DATASET_FOLDER"
fi
echo ""

# Check if using direct hypothesis mode
if [ "$HYPOTHESIS_MODE" = "direct hypothesis" ]; then
    echo "🎯 Running with direct hypothesis (single call)..."
    echo "=========================================="
    
    # Run the discovery agent with direct hypothesis (single call)
    if [ "$VERBOSE" = "true" ]; then
        python discover.py \
            --model-name "$MODEL_NAME" \
            --device "$DEVICE" \
            --max-tokens "$MAX_TOKENS" \
            --max-conversation-length "$MAX_CONVERSATION_LENGTH" \
            --auto-approve-after "$AUTO_APPROVE_AFTER" \
            --outer-loop-iterations "$OUTER_LOOP_ITERATIONS" \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --hypothesis "$HYPOTHESIS" \
            --dataset-folder "$DATASET_FOLDER" \
            --verbose
    else
        python discover.py \
            --model-name "$MODEL_NAME" \
            --device "$DEVICE" \
            --max-tokens "$MAX_TOKENS" \
            --max-conversation-length "$MAX_CONVERSATION_LENGTH" \
            --auto-approve-after "$AUTO_APPROVE_AFTER" \
            --outer-loop-iterations "$OUTER_LOOP_ITERATIONS" \
            --hypothesis-mode "$HYPOTHESIS_MODE" \
            --hypothesis "$HYPOTHESIS" \
            --dataset-folder "$DATASET_FOLDER"
    fi
    
    echo "✅ Completed direct hypothesis run"
else
    # Loop through 2 and 3 keywords for generate/direct modes
    for NUM_KEYWORDS in 2 3; do
        echo "🔄 Running with $NUM_KEYWORDS keywords..."
        echo "=========================================="
        
        # Run the discovery agent with direct arguments
        if [ "$VERBOSE" = "true" ]; then
            python discover.py \
                --model-name "$MODEL_NAME" \
                --device "$DEVICE" \
                --max-tokens "$MAX_TOKENS" \
                --max-conversation-length "$MAX_CONVERSATION_LENGTH" \
                --auto-approve-after "$AUTO_APPROVE_AFTER" \
                --outer-loop-iterations "$OUTER_LOOP_ITERATIONS" \
                --hypothesis-mode "$HYPOTHESIS_MODE" \
                --keywords-file "$KEYWORDS_FILE" \
                --num-keywords "$NUM_KEYWORDS" \
                --dataset-folder "$DATASET_FOLDER" \
                --num-sampling-runs "$NUM_SAMPLING_RUNS" \
                --verbose
        else
            python discover.py \
                --model-name "$MODEL_NAME" \
                --device "$DEVICE" \
                --max-tokens "$MAX_TOKENS" \
                --max-conversation-length "$MAX_CONVERSATION_LENGTH" \
                --auto-approve-after "$AUTO_APPROVE_AFTER" \
                --outer-loop-iterations "$OUTER_LOOP_ITERATIONS" \
                --hypothesis-mode "$HYPOTHESIS_MODE" \
                --keywords-file "$KEYWORDS_FILE" \
                --num-keywords "$NUM_KEYWORDS" \
                --dataset-folder "$DATASET_FOLDER" \
                --num-sampling-runs "$NUM_SAMPLING_RUNS"
        fi
        
        echo "✅ Completed run with $NUM_KEYWORDS keywords"
        echo ""
    done
fi

echo "🎉 All runs completed!"
    