# Configuration Guide

This guide will help you configure DiscoADRD_v3 for your environment.

## Quick Setup Checklist

1. ? Update dataset paths in `config.yml`
2. ? Update paths in `run.sh` (if using HPC)
3. ? Set up HuggingFace cache directory (optional)
4. ? Verify all dependencies are installed

## Configuration Files

### 1. `config.yml` - Main Configuration File

The main configuration file is `config.yml`. Key settings to update:

#### Dataset Paths
```yaml
paths:
  dataset_folder: "/path/to/your/NACC/datasets"  # Update this to your dataset location
```

#### CSV Batch File (if using csv_batch mode)
```yaml
experiment:
  csv_file: "batch.csv"  # Update to your batch file path (relative or absolute)
```

#### Working Directory
```yaml
paths:
  working_dir: "working"  # Temporary working directory (auto-created)
```

### 2. `run.sh` - HPC Job Submission Script

If you're using HPC (SGE/PBS), update the following in `run.sh`:

#### SGE Directives
```bash
#$ -P your_project_name      # Update to your project name
#$ -o analysis/qsub_runs     # Output directory (relative to working directory)
```

#### Working Directory
The script automatically changes to the script directory using:
```bash
cd "$(dirname "$0")"
```

If you need a different working directory, update this line.

### 3. Environment Variables

#### HuggingFace Cache (Optional)
If you want to use a custom HuggingFace cache directory:

**In shell scripts:**
```bash
export HF_HOME="/path/to/your/huggingface/cache"
```

**In Python code:**
```python
import os
os.environ["HF_HOME"] = "/path/to/your/huggingface/cache"
```

**Default location:** `~/.cache/huggingface`

## Path Configuration Examples

### Example 1: Local Installation
```yaml
paths:
  dataset_folder: "/home/user/data/NACC"
  working_dir: "working"
```

### Example 2: HPC Cluster
```yaml
paths:
  dataset_folder: "/project/your_project/datasets/NACC"
  working_dir: "working_${USER}"
```

### Example 3: Relative Paths
```yaml
paths:
  dataset_folder: "../datasets/NACC"  # Relative to script location
  working_dir: "working"
```

## Model Configuration

### Default Model
The default model is `Qwen/Qwen3-4B-Instruct-2507`. To use a different model:

```yaml
model:
  name: "your-model-name"
  device: "auto"  # auto, cuda, cpu
```

### vLLM Configuration
If using vLLM backend:

```yaml
model:
  client_type: "vllm"  # or "hf" for HuggingFace
  max_model_len: 224848
```

## Verification

After configuration, verify your setup:

1. **Check paths exist:**
   ```bash
   ls -la /path/to/your/NACC/datasets
   ```

2. **Test configuration:**
   ```bash
   python discover.py --help
   ```

3. **Dry run (if supported):**
   ```bash
   python discover.py --verbose --hypothesis-mode "direct" --dataset-folder "/path/to/test"
   ```

## Common Issues

### Issue: "Dataset folder not found"
**Solution:** Verify the path in `config.yml` is correct and accessible.

### Issue: "Permission denied"
**Solution:** Check file permissions on dataset folder and working directory.

### Issue: "CUDA out of memory"
**Solution:** Reduce `max_tokens` or `max_model_len` in `config.yml`, or use CPU mode.

### Issue: "Model not found"
**Solution:** Check internet connection for HuggingFace downloads, or verify model name.

## Advanced Configuration

For advanced configuration options, see the main README.md file or check the `Config` class in `discovery_agent_pkg/core/config.py`.

---

*For more information, see the main [README.md](README.md)*
