## Agentic AI for automated hypothesis testing in Alzheimer’s disease and related dementias

## 🎯 Overview

DiscoADRD employs a sophisticated multi-agent architecture where specialized AI agents (Scientist, Critic, and Planning agents) collaborate to:
- Analyze complex medical datasets
- Generate and test scientific hypotheses
- Perform comprehensive statistical analysis
- Generate research insights and findings
- Automate the scientific discovery workflow

## 📁 Repository Structure

```
DiscoADRD/
├── DiscoADRD_v3/              # Latest version (recommended)
│   ├── discover.py            # Main entry point
│   ├── config.yml             # Configuration file
│   ├── run.sh                 # HPC job submission script
│   ├── batch.csv              # Sample batch hypotheses file
│   └── discovery_agent_pkg/   # Core package
│       ├── agents/            # AI agent implementations
│       ├── core/              # Core system components
│       ├── data/              # Data processing modules
│       ├── execution/         # Code execution system
│       ├── experiment/        # Experiment management
│       └── ui/                # User interface components
│
├── vLLM/                      # vLLM inference client utilities
│   ├── client.py              # vLLM multi-GPU client
│   ├── discovery_agent_client.py
│   └── run.sh                 # vLLM runner script
│
├── LICENSE                    # License file
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Access to NACC datasets
- HPC environment with SGE/PBS support (optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DiscoADRD
   ```

2. **Set up environment** (for HPC environments):
   ```bash
   module load miniconda
   module load cuda
   conda activate llms
   ```

3. **Install dependencies**:
   ```bash
   pip install transformers torch huggingface-hub python-dotenv
   pip install pandas numpy scipy scikit-learn matplotlib seaborn
   pip install pyyaml tqdm vllm
   ```

4. **Configure the system**:
   - Edit `DiscoADRD_v3/config.yml` with your dataset paths and settings
   - Update paths in `run.sh` scripts to match your environment

### Basic Usage

```bash
cd DiscoADRD_v3
python discover.py
```

For detailed usage instructions, see [DiscoADRD_v3/README.md](DiscoADRD_v3/README.md).

## 📚 Documentation

- **[DiscoADRD_v3/README.md](DiscoADRD_v3/README.md)** - Comprehensive documentation for the latest version
- **[discovery_agent_pkg/README.md](DiscoADRD_v3/discovery_agent_pkg/README.md)** - Package-level documentation

## 🔧 Key Features

- **Multi-Agent Architecture**: Scientist, Critic, and Planning agents work collaboratively
- **Advanced Statistical Analysis**: Comprehensive statistical testing including regression, survival analysis, and ML
- **NACC Dataset Integration**: Specialized for Alzheimer's disease research
- **Automated Code Execution**: Safe Python code execution with timeout protection
- **Multiple Hypothesis Modes**: Direct, batch, and keyword-based hypothesis generation
- **HPC Integration**: Designed for high-performance computing environments

## 🏗️ Architecture

The system uses a modular architecture with clear separation of concerns:

- **Agents**: Specialized AI agents for different roles (Scientist, Critic, Planner)
- **Core**: Main orchestration and configuration management
- **Data**: Data processing, caching, and file management
- **Execution**: Safe code execution with sandboxing
- **Experiment**: Experiment lifecycle management
- **UI**: User interface and progress tracking

## 📊 Supported Workflows

1. **Direct Hypothesis Testing**: Test a specific hypothesis directly
2. **CSV Batch Processing**: Process multiple hypotheses from a CSV file
3. **Keyword-Based Generation**: Generate hypotheses from keyword sampling
4. **Direct Mode**: Process without hypothesis generation

## 🔬 Research Applications

DiscoADRD is specifically designed for:
- Alzheimer's Disease and Related Dementias (ADRD) research
- NACC dataset analysis
- Automated hypothesis generation and testing
- Clinical research data analysis
- Epidemiological studies

## 📝 Configuration

Configuration is managed through:
- **YAML files**: `config.yml` for system-wide settings
- **Command-line arguments**: Override YAML settings
- **Environment variables**: For HPC and system-specific settings

See `DiscoADRD_v3/config.yml` for detailed configuration options.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) for license information.

## 📞 Support

For questions and support:
- Check the documentation in `DiscoADRD_v3/README.md`
- Review configuration options in `config.yml`
- Examine verbose output logs
- Contact the development team

## 🔄 Version Information

- **DiscoADRD_v3**: Latest version with enhanced features (recommended)
- **vLLM**: Utilities for vLLM inference client

---

*Last updated: January 2025*
