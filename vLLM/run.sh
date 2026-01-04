# Change to the vLLM directory
cd "$(dirname "$0")"  # Automatically change to script directory
module load miniconda
module load cuda
conda activate llms
python client.py