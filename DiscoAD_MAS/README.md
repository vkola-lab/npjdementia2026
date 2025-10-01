# DiscoAD_MAS with Planner Refinement

A comprehensive AI-powered scientific discovery system for Alzheimer's Disease and Related Dementias (ADRD) research. This system uses multi-agent collaboration between Scientist and Critic agents to analyze NACC (National Alzheimer's Coordinating Center) datasets and generate insights through iterative conversation and hypothesis testing.

## 🚀 Overview

DiscoAD_MAS (Discovery Agent for ADRD with Multi-Agent System) is an advanced research tool that combines:
- **Multi-Agent Architecture**: Scientist and Critic agents collaborate to analyze data
- **Planner Refinement**: Iterative planning and refinement of research hypotheses
- **NACC Dataset Integration**: Specialized for Alzheimer's disease research datasets
- **Automated Code Execution**: Safe Python code execution for data analysis
- **Keyword-Based Sampling**: Intelligent sampling of research variables

## 🏗️ Architecture

The system consists of several key components:

- **DiscoveryAgentSystem**: Main orchestrator that coordinates all agents
- **ModelClient**: Interface to HuggingFace models (default: Qwen/Qwen3-4B-Instruct-2507)
- **ConversationManager**: Manages multi-turn conversations between agents
- **CodeExecutor**: Safely executes Python code for data analysis
- **UI**: User interface for interaction and monitoring
- **Config**: Configuration management for all system parameters

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Access to NACC datasets
- Required Python packages (see installation section)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DiscoAD_MAS_with_planner_refinement
   ```

2. **Set up environment**:
   ```bash
   module load miniconda
   module load cuda
   conda activate llms
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # If available
   # Or install manually:
   pip install transformers torch huggingface-hub python-dotenv
   ```

## 🎯 Usage

### Quick Start

Run the discovery agent with default settings:
```bash
python discover.py
```

### Advanced Configuration

Use the provided shell script for comprehensive runs:
```bash
./run.sh
```

### Command Line Options

The system supports extensive configuration through command line arguments:

```bash
python discover.py \
    --model-name "Qwen/Qwen3-4B-Instruct-2507" \
    --device "auto" \
    --max-tokens 16384 \
    --max-conversation-length 7 \
    --auto-approve-after 1 \
    --outer-loop-iterations 2 \
    --hypothesis-mode "generate" \
    --keywords-file "adrd_keywords.txt" \
    --num-keywords 3 \
    --dataset-folder "/path/to/NACC/datasets" \
    --num-sampling-runs 10 \
    --verbose
```

### Key Parameters

- **`--model-name`**: HuggingFace model to use (default: Qwen/Qwen3-4B-Instruct-2507)
- **`--hypothesis-mode`**: "generate" for multiple hypotheses, "direct" for single input
- **`--num-keywords`**: Number of keywords to sample per run (2 or 3 recommended)
- **`--num-sampling-runs`**: Number of keyword sampling iterations
- **`--outer-loop-iterations`**: Number of data prep → planner → scientist/critic cycles
- **`--dataset-folder`**: Path to NACC dataset folder
- **`--verbose`**: Enable detailed output logging

## 📊 Dataset Integration

The system is designed to work with NACC datasets containing:

- **Demographics**: age, education, gender, race, ethnicity, marital status
- **Cognitive Tests**: MMSE, MoCA, CDR, memory, attention, executive function
- **Clinical Variables**: Various ADRD-related clinical measurements

Keywords are defined in `adrd_keywords.txt` and can be customized for specific research needs.

## 🔄 Workflow

1. **Initialization**: System loads configuration and initializes agents
2. **Keyword Sampling**: Randomly samples keywords from the predefined list
3. **Data Preparation**: Loads and prepares relevant dataset subsets
4. **Hypothesis Generation**: Scientist agent generates research hypotheses
5. **Planning**: Planner refines and structures the research approach
6. **Analysis**: Scientist and Critic agents collaborate on data analysis
7. **Iteration**: Process repeats with refined hypotheses and approaches
8. **Output**: Results are saved to designated output directories

## 📁 Directory Structure

```
DiscoAD_MAS_with_planner_refinement/
├── discover.py                 # Main entry point
├── run.sh                     # Comprehensive run script
├── adrd_keywords.txt          # Research keywords for sampling
├── discovery_agent_pkg/       # Core package
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── discovery_agent.py    # Main agent system
│   ├── models.py             # Model interfaces
│   ├── conversation.py       # Conversation management
│   ├── code_executor.py      # Code execution
│   ├── ui.py                 # User interface
│   └── utils.py              # Utility functions
├── outputs/                  # Analysis results
├── results/                  # Final outputs
├── responses/                # Agent responses
├── qsub_runs/               # Job submission logs
└── working/                 # Temporary working files
```

## 🎛️ Configuration

The system can be configured through:

1. **Command line arguments** (highest priority)
2. **Environment variables**
3. **Configuration files**
4. **Default values**

Key configuration areas:
- Model selection and parameters
- Agent behavior settings
- Dataset paths and processing
- Output verbosity and logging
- Resource allocation (GPU, memory)

## 🚀 HPC Integration

The system is designed for High Performance Computing environments:

- **SGE/PBS Integration**: Use `run.sh` for job submission
- **GPU Support**: Automatic CUDA detection and utilization
- **Resource Management**: Configurable memory and compute requirements
- **Batch Processing**: Support for multiple parallel runs

Example SGE submission:
```bash
qsub run.sh
```

## 📈 Output and Results

Results are organized in several directories:

- **`outputs/`**: Intermediate analysis results
- **`results/`**: Final research findings and summaries
- **`responses/`**: Detailed agent conversation logs
- **`qsub_runs/`**: Job execution logs and monitoring

## 🔧 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure CUDA is properly loaded and GPU is available
2. **Model Loading**: Check internet connection for HuggingFace model downloads
3. **Dataset Access**: Verify dataset folder paths and permissions
4. **Memory Issues**: Reduce `--max-tokens` or `--num-sampling-runs`

### Debug Mode

Enable verbose output for detailed debugging:
```bash
python discover.py --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Add your license information here]

## 📞 Support

For questions and support:
- Check the troubleshooting section
- Review the configuration options
- Examine the verbose output logs
- Contact the development team

## 🔬 Research Applications

This system is specifically designed for:
- Alzheimer's disease research
- Dementia-related data analysis
- Multi-agent scientific discovery
- Hypothesis generation and testing
- Automated research workflow optimization

---

*Last updated: [Current Date]*