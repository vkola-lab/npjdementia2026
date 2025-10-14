# DiscoAD_MAS - Multi-Agent Scientific Discovery System

A comprehensive AI-powered scientific discovery system for Alzheimer's Disease and Related Dementias (ADRD) research. This system uses advanced multi-agent collaboration between Scientist, Critic, and Planning agents to analyze NACC (National Alzheimer's Coordinating Center) datasets and generate insights through iterative conversation, hypothesis testing, and automated statistical analysis.

## 🚀 Overview

DiscoAD_MAS (Discovery Agent for ADRD with Multi-Agent System) is an advanced research tool that combines:
- **Multi-Agent Architecture**: Scientist, Critic, and Enhanced Planning agents collaborate to analyze data
- **Advanced Statistical Analysis**: Comprehensive statistical testing including regression, survival analysis, and machine learning
- **NACC Dataset Integration**: Specialized for Alzheimer's disease research datasets with intelligent variable selection
- **Automated Code Execution**: Safe Python code execution with timeout protection and persistent namespaces
- **Multiple Hypothesis Modes**: Support for direct hypothesis, CSV batch processing, and keyword-based generation
- **Intelligent Data Processing**: LLM-powered variable selection and preprocessing with missing data handling
- **HPC Integration**: Designed for high-performance computing environments with SGE/PBS job submission

## 🏗️ Architecture

The system consists of several key components organized in a modular structure:

### Core Components
- **DiscoveryAgentSystem**: Main orchestrator that coordinates all agents and manages the overall workflow
- **Config**: Advanced configuration management with YAML file support and command-line overrides
- **ModelClient**: Interface to HuggingFace models (default: Qwen/Qwen3-4B-Instruct-2507) with automatic device detection

### Agent System
- **Scientist Agent**: Performs comprehensive statistical analysis including EDA, hypothesis testing, regression, survival analysis, and machine learning
- **Critic Agent**: Reviews and validates scientific approaches, provides feedback on methodology and results
- **Enhanced Planning Agent**: Designs multi-step research strategies and analytical frameworks (optional, can be disabled)

### Data Processing
- **DataCache**: Centralized data management with CSV and JSON file caching
- **SimplifiedDataProcessor**: LLM-powered variable selection and intelligent data preprocessing
- **FileManager**: Handles file operations and output management

### Execution & Analysis
- **CodeExecutor**: Safe Python code execution with timeout protection and persistent global namespaces
- **ConversationManager**: Manages multi-turn conversations between agents with context preservation
- **ExperimentRunner**: Orchestrates sampling runs, hypothesis processing, and outer loop iterations

### User Interface
- **UI Interface**: Rich console interface with colored output, progress tracking, and detailed logging
- **Progress Tracking**: Real-time progress indicators and status updates

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- Access to NACC datasets
- HPC environment with SGE/PBS support (optional, for cluster deployment)
- Required Python packages (see installation section)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DiscoAD_MAS
   ```

2. **Set up environment** (for HPC environments):
   ```bash
   module load miniconda
   module load cuda
   conda activate llms
   ```

3. **Install dependencies**:
   ```bash
   # Core dependencies
   pip install transformers torch huggingface-hub python-dotenv
   
   # Data processing dependencies
   pip install pandas numpy scipy scikit-learn matplotlib seaborn
   
   # Additional utilities
   pip install pyyaml tqdm
   ```

4. **Verify installation**:
   ```bash
   python discover.py --help
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

### Hypothesis Modes

The system supports four different hypothesis processing modes:

#### 1. Direct Hypothesis Mode
Test a single, specific hypothesis:
```bash
python discover.py \
    --hypothesis-mode "direct hypothesis" \
    --hypothesis "Greater years of education are linked to lower dementia incidence (cognitive reserve)." \
    --dataset-folder "/path/to/NACC/datasets"
```

#### 2. CSV Batch Mode
Process multiple hypotheses from a CSV file:
```bash
python discover.py \
    --hypothesis-mode "csv_batch" \
    --csv-file "sample_hypotheses.csv" \
    --dataset-folder "/path/to/NACC/datasets"
```

#### 3. Generate Mode
Generate hypotheses based on keyword sampling:
```bash
python discover.py \
    --hypothesis-mode "generate" \
    --dataset-folder "/path/to/NACC/datasets" \
    --num-keywords 3 \
    --num-sampling-runs 10
```

#### 4. Direct Mode
Use direct input without hypothesis generation:
```bash
python discover.py \
    --hypothesis-mode "direct" \
    --dataset-folder "/path/to/NACC/datasets"
```

### Configuration Options

The system supports extensive configuration through command line arguments and YAML files:

```bash
python discover.py \
    --hypothesis-mode "direct hypothesis" \
    --hypothesis "Your hypothesis here" \
    --dataset-folder "/path/to/NACC/datasets" \
    --verbose \
    --disable-planning-agent
```

### Key Parameters

- **`--hypothesis-mode`**: Choose from "generate", "direct", "direct hypothesis", or "csv_batch"
- **`--hypothesis`**: Direct hypothesis text (required for "direct hypothesis" mode)
- **`--csv-file`**: Path to CSV file with hypotheses (required for "csv_batch" mode)
- **`--dataset-folder`**: Path to NACC dataset folder
- **`--verbose`**: Enable detailed output logging
- **`--disable-planning-agent`**: Disable the enhanced planning agent
- **`--config`**: Path to custom YAML configuration file

## 📊 Dataset Integration

The system is designed to work with NACC datasets containing:

- **Demographics**: age, education, gender, race, ethnicity, marital status
- **Cognitive Tests**: MMSE, MoCA, CDR, memory, attention, executive function
- **Clinical Variables**: Various ADRD-related clinical measurements

Keywords are defined in `adrd_keywords.txt` and can be customized for specific research needs.

## 🔄 Workflow

The system follows a sophisticated multi-stage workflow:

### 1. Initialization Phase
- Load configuration from YAML file and command-line arguments
- Initialize model client with automatic device detection (CUDA/CPU)
- Set up data cache and file management systems
- Create agent instances (Scientist, Critic, Enhanced Planning Agent)

### 2. Hypothesis Processing Phase
- **Direct Hypothesis Mode**: Use provided hypothesis directly
- **CSV Batch Mode**: Load and process multiple hypotheses from CSV file
- **Generate Mode**: Sample keywords and generate hypotheses using LLM
- **Direct Mode**: Process without hypothesis generation

### 3. Data Processing Phase
- Initialize data cache with CSV and JSON files from dataset folder
- LLM-powered variable selection based on hypothesis relevance
- Intelligent data preprocessing with missing value handling
- Categorical variable encoding and data quality assessment
- Save processed dataset to working directory

### 4. Planning Phase (if enabled)
- Enhanced Planning Agent designs comprehensive research strategy
- Multi-step analytical framework development
- Statistical methodology recommendations
- Validation and robustness planning

### 5. Analysis Phase
- **First Turn**: Comprehensive Exploratory Data Analysis (EDA)
- **Subsequent Turns**: Hypothesis-specific statistical testing
- Advanced statistical analysis including:
  - Regression modeling (linear, logistic, survival)
  - Machine learning approaches
  - Mediation and moderation analysis
  - Subgroup analyses and heterogeneity assessment
- Code execution with timeout protection and persistent namespaces

### 6. Iteration Phase
- Critic agent reviews and validates approaches
- Scientist agent refines analysis based on feedback
- Outer loop iterations for comprehensive exploration
- Previous iteration context integration

### 7. Output Phase
- Save detailed conversation logs
- Generate hypothesis approval tracking reports
- Create analysis results and summaries
- Export findings to designated directories

## 📁 Directory Structure

```
DiscoAD_MAS/
├── discover.py                    # Main entry point
├── run.sh                        # Comprehensive HPC run script
├── config.yml                    # YAML configuration file
├── adrd_keywords.txt             # Research keywords for sampling
├── sample_hypotheses.csv         # Example CSV file for batch processing
├── todo.txt                      # Development notes and future improvements
├── discovery_agent_pkg/          # Core package
│   ├── __init__.py              # Package initialization with lazy imports
│   ├── main.py                  # Main execution module
│   ├── discovery_agent.py       # Main agent system
│   ├── agents/                  # Agent implementations
│   │   ├── __init__.py
│   │   ├── scientist.py         # Scientist agent with advanced statistical capabilities
│   │   ├── critic.py            # Critic agent for validation
│   │   ├── planner.py           # Enhanced planning agent
│   │   └── models.py            # Agent model interfaces
│   ├── core/                    # Core system components
│   │   ├── __init__.py
│   │   ├── agent.py             # Main agent system implementation
│   │   ├── config.py            # Advanced configuration management
│   │   └── utils.py             # Core utilities
│   ├── data/                    # Data processing components
│   │   ├── __init__.py
│   │   ├── cache.py             # Data caching system
│   │   ├── processor.py         # Simplified data processor
│   │   └── file_manager.py      # File management utilities
│   ├── execution/               # Code execution system
│   │   ├── __init__.py
│   │   ├── executor.py          # Safe code execution with timeout
│   │   └── utils.py             # Execution utilities
│   ├── experiment/              # Experiment management
│   │   ├── __init__.py
│   │   ├── runner.py            # Experiment orchestration
│   │   ├── conversation.py      # Conversation management
│   │   └── hypothesis.py        # Hypothesis processing
│   └── ui/                      # User interface components
│       ├── __init__.py
│       ├── interface.py         # Main UI interface
│       ├── colors.py            # Color schemes
│       ├── formatting.py        # Text formatting
│       └── progress.py          # Progress tracking
├── analysis/                    # Analysis outputs
│   ├── results/                # Final analysis results and tracking files
│   ├── responses/              # Detailed agent conversation logs
│   ├── qsub_runs/             # HPC job submission logs
│   └── figures/               # Generated figures and visualizations
├── working/                    # Temporary working files (auto-created/deleted)
└── __pycache__/               # Python cache files
```

## 🎛️ Configuration

The system supports flexible configuration through multiple layers:

### Configuration Priority (highest to lowest)
1. **Command line arguments** (highest priority)
2. **YAML configuration file** (`config.yml`)
3. **Default values**

### YAML Configuration File

The `config.yml` file provides comprehensive system configuration:

```yaml
# Model Configuration
model:
  name: "Qwen/Qwen3-4B-Instruct-2507"
  device: "auto"  # auto, cuda, cpu
  max_tokens: 16384

# Agent Configuration
agents:
  max_conversation_length: 7
  auto_approve_after_attempts: 4
  code_timeout: 30  # seconds
  enable_planning_agent: false
  enable_code_execution: true

# Experiment Configuration
experiment:
  hypothesis_mode: "direct hypothesis"
  hypothesis: "Greater years of education are linked to lower dementia incidence (cognitive reserve)."
  csv_file: ""  # Path to CSV file for batch mode
  num_keywords: 3
  num_sampling_runs: 10
  outer_loop_iterations: 2

# File Paths
paths:
  keywords_file: "adrd_keywords.txt"
  analysis_dir: "analysis"
  responses_dir: "analysis/responses"
  results_dir: "analysis/results"
  qsub_runs_dir: "analysis/qsub_runs"
  working_dir: "working"

# Output Configuration
output:
  verbose: true
  save_responses: true
  save_results: true
```

### Key Configuration Areas
- **Model Selection**: HuggingFace model name, device selection, token limits
- **Agent Behavior**: Conversation length, approval thresholds, code execution settings
- **Experiment Settings**: Hypothesis modes, sampling parameters, iteration counts
- **File Paths**: Directory structure and file locations
- **Output Control**: Verbosity levels, response saving, result generation

## ✨ Key Features

### Advanced Statistical Analysis
- **Comprehensive Statistical Testing**: t-tests, ANOVA, chi-square tests, normality tests
- **Regression Analysis**: Linear, logistic, and survival regression models
- **Machine Learning**: Classification, clustering, and dimensionality reduction
- **Advanced Analytics**: Mediation, moderation, interaction effects, subgroup analysis
- **Effect Sizes & Power**: Cohen's d, Cramér's V, eta-squared, power analysis

### Intelligent Data Processing
- **LLM-Powered Variable Selection**: Automatic selection of relevant variables based on hypothesis
- **Missing Data Handling**: Intelligent missing value detection and handling strategies
- **Data Quality Assessment**: Comprehensive data quality evaluation and reporting
- **Categorical Encoding**: Automatic dummy variable creation for categorical data
- **Data Caching**: Efficient caching system for large datasets

### Multi-Agent Collaboration
- **Enhanced Planning Agent**: Designs comprehensive research strategies and analytical frameworks
- **Scientist Agent**: Performs advanced statistical analysis with comprehensive capabilities
- **Critic Agent**: Validates approaches and provides methodological feedback
- **Iterative Refinement**: Multi-turn conversations with context preservation

### Flexible Hypothesis Processing
- **Direct Hypothesis Mode**: Test specific hypotheses directly
- **CSV Batch Mode**: Process multiple hypotheses from CSV files
- **Generate Mode**: Automatic hypothesis generation from keyword sampling
- **Direct Mode**: Process without hypothesis generation

### Safe Code Execution
- **Timeout Protection**: Prevents infinite loops and long-running code
- **Persistent Namespaces**: Maintains variable state across code blocks
- **Error Handling**: Comprehensive error capture and reporting
- **Sandboxed Environment**: Safe execution of LLM-generated code

### HPC Integration
- **SGE/PBS Support**: Native job submission and management
- **GPU Optimization**: Automatic CUDA detection and utilization
- **Resource Management**: Configurable memory and compute requirements
- **Batch Processing**: Support for multiple parallel runs

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

The system generates comprehensive outputs organized in structured directories:

### Output Directory Structure

- **`analysis/results/`**: Final analysis results and hypothesis tracking files
  - `hypothesis_approval_tracking_*.csv`: Detailed tracking of hypothesis processing and approval status
  - Contains timestamps, model information, and processing statistics

- **`analysis/responses/`**: Detailed agent conversation logs
  - `session_log_*.txt`: Complete conversation transcripts between agents
  - Includes all agent interactions, code execution results, and analysis outputs

- **`analysis/qsub_runs/`**: HPC job submission and execution logs
  - SGE/PBS job output files
  - System resource usage and performance metrics

- **`analysis/figures/`**: Generated visualizations and plots
  - Statistical plots, charts, and analysis visualizations
  - High-quality figures for research publications

- **`working/`**: Temporary working files (auto-created and cleaned up)
  - Processed datasets and intermediate analysis files
  - Automatically managed during execution

### Output File Naming Convention

Files are named with descriptive timestamps and configuration details:
- **Results**: `hypothesis_approval_tracking_model_{model}_mode_{mode}_kw_{keywords}_runs_{runs}_outer_{iterations}_auto_{approval}_{timestamp}.csv`
- **Responses**: `session_log_{timestamp}.txt`
- **Qsub runs**: Standard SGE/PBS output format

### Key Output Information

- **Hypothesis Tracking**: Complete record of hypothesis processing, approval status, and iteration details
- **Conversation Logs**: Full agent interactions with code execution results and analysis outputs
- **Statistical Results**: Comprehensive statistical analysis results with p-values, effect sizes, and confidence intervals
- **Performance Metrics**: Execution time, resource usage, and system performance data

## 🔧 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: 
   - Ensure CUDA is properly loaded: `module load cuda`
   - Check GPU availability: `nvidia-smi`
   - Verify GPU memory: `--device "auto"` will automatically detect best device

2. **Model Loading Issues**:
   - Check internet connection for HuggingFace model downloads
   - Verify model name: `Qwen/Qwen3-4B-Instruct-2507`
   - Check available disk space for model caching

3. **Dataset Access Issues**:
   - Verify dataset folder paths and permissions
   - Ensure CSV and JSON files are present in dataset folder
   - Check file formats and encoding

4. **Memory Issues**:
   - Reduce `max_tokens` in config.yml or use `--max-tokens` parameter
   - Decrease `num_sampling_runs` for fewer iterations
   - Use `--disable-planning-agent` to reduce memory usage

5. **Configuration Issues**:
   - Verify `config.yml` syntax and file permissions
   - Check command-line argument format
   - Ensure required parameters are provided for specific modes

6. **Code Execution Issues**:
   - Check code timeout settings (default: 30 seconds)
   - Verify Python dependencies are installed
   - Review error messages in conversation logs

### Debug Mode

Enable verbose output for detailed debugging:
```bash
python discover.py --verbose
```

### Log Analysis

Check the following files for detailed error information:
- **Conversation logs**: `analysis/responses/session_log_*.txt`
- **HPC job logs**: `analysis/qsub_runs/`
- **System output**: Console output with `--verbose` flag

### Performance Optimization

- **GPU Utilization**: Use `--device "cuda"` for GPU acceleration
- **Memory Management**: Adjust `max_tokens` based on available memory
- **Batch Processing**: Use CSV batch mode for multiple hypotheses
- **Resource Allocation**: Configure HPC resources in `run.sh`

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
- **Alzheimer's Disease Research**: Comprehensive analysis of ADRD datasets with specialized variable selection
- **Dementia-Related Data Analysis**: Advanced statistical analysis of cognitive and clinical variables
- **Multi-Agent Scientific Discovery**: Collaborative AI agents for hypothesis testing and validation
- **Hypothesis Generation and Testing**: Automated hypothesis generation from keyword sampling
- **Automated Research Workflow Optimization**: Streamlined research processes with intelligent data processing
- **Clinical Research**: Support for clinical trial data analysis and biomarker research
- **Epidemiological Studies**: Large-scale population health research with NACC datasets

## 📚 Additional Resources

- **Sample Data**: Use `sample_hypotheses.csv` as a template for batch processing
- **Keywords**: Customize `adrd_keywords.txt` for domain-specific research
- **Configuration**: Modify `config.yml` for system-wide settings
- **HPC Scripts**: Use `run.sh` for cluster deployment

## 🔄 Version History

- **v1.0.0**: Initial release with multi-agent system
- **Current**: Enhanced planning agent, CSV batch processing, advanced statistical analysis

---

*Last updated: January 2025*