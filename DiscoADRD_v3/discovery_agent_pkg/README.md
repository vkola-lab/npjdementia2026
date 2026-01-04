# Discovery Agent Package

A comprehensive AI-powered scientific discovery system that uses multiple agents (Scientist, Critic, Planning Agent) to test hypotheses and analyze data.

## Package Structure

The package has been reorganized into a clear, modular structure:

```
discovery_agent_pkg/
├── __init__.py                 # Main package initialization
├── main.py                     # Entry point for running the system
├── README.md                   # This documentation
│
├── core/                       # Core system components
│   ├── __init__.py
│   ├── agent.py               # Main DiscoveryAgentSystem class
│   ├── config.py              # Configuration management
│   └── utils.py               # Core utility functions
│
├── agents/                     # AI Agent implementations
│   ├── __init__.py
│   ├── models.py              # ModelClient and base Agent class
│   ├── scientist.py           # Scientist agent implementation
│   ├── critic.py              # Critic agent implementation
│   └── planner.py             # Planning agent implementation
│
├── data/                       # Data handling components
│   ├── __init__.py
│   ├── cache.py               # Data caching system
│   ├── processor.py           # Simplified data processing
│   └── file_manager.py        # File management utilities
│
├── experiment/                 # Experiment execution
│   ├── __init__.py
│   ├── runner.py              # Main experiment runner
│   ├── hypothesis.py          # Hypothesis management
│   └── conversation.py        # Conversation management
│
├── execution/                  # Code execution utilities
│   ├── __init__.py
│   ├── executor.py            # Code executor with timeout
│   └── utils.py               # Execution utilities
│
└── ui/                         # User interface
    ├── __init__.py
    └── interface.py           # UI components and display
```

## Key Components

### Core Module (`core/`)
- **`agent.py`**: Main `DiscoveryAgentSystem` class that orchestrates the entire discovery process
- **`config.py`**: Configuration management and command-line argument parsing
- **`utils.py`**: Core utility functions for GPU cleanup, signal handling, and logging

### Agents Module (`agents/`)
- **`models.py`**: `ModelClient` for HuggingFace model interaction and base `Agent` class
- **`scientist.py`**: Scientist agent for generating and testing hypotheses
- **`critic.py`**: Critic agent for reviewing and approving results
- **`planner.py`**: Planning agent for strategic analysis planning

### Data Module (`data/`)
- **`cache.py`**: Centralized data cache for CSV and JSON data
- **`processor.py`**: Simplified data processing pipeline
- **`file_manager.py`**: File operations and directory management

### Experiment Module (`experiment/`)
- **`runner.py`**: Main experiment execution and orchestration
- **`hypothesis.py`**: Hypothesis generation and management
- **`conversation.py`**: Conversation history and context management

### Execution Module (`execution/`)
- **`executor.py`**: Robust Python code executor with timeout mechanisms
- **`utils.py`**: Code execution utilities and helpers

### UI Module (`ui/`)
- **`interface.py`**: User interface components with colored output and logging

## Usage

### Basic Usage

```python
from discovery_agent_pkg import DiscoveryAgentSystem, Config

# Create configuration
config = Config()
config.dataset_folder = "/path/to/your/NACC/dataset"
config.model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Initialize and run the discovery agent
agent = DiscoveryAgentSystem(config)
agent.run()
```

### Command Line Usage

```bash
python discover.py --dataset-folder /path/to/your/NACC/dataset --model-name Qwen/Qwen3-4B-Instruct-2507 --verbose
```

### Advanced Usage

```python
from discovery_agent_pkg import DiscoveryAgentSystem, Config
from discovery_agent_pkg.agents.models import ModelClient
from discovery_agent_pkg.experiment.runner import ExperimentRunner

# Custom configuration
config = Config()
config.dataset_folder = "/path/to/your/NACC/dataset"
config.model_name = "Qwen/Qwen3-4B-Instruct-2507"
config.verbose = True
config.auto_approve_after_attempts = 5

# Initialize components
model_client = ModelClient(config)
agent = DiscoveryAgentSystem(config)

# Run with custom parameters
agent.run()
```

## Configuration Options

The system supports various configuration options:

- **`dataset_folder`**: Path to folder containing NACC CSV and JSON files
- **`model_name`**: HuggingFace model name to use (default: "Qwen/Qwen3-4B-Instruct-2507")
- **`verbose`**: Enable verbose output
- **`auto_approve_after_attempts`**: Number of attempts before auto-approval
- **`outer_loop_iterations`**: Number of outer loop iterations
- **`hypothesis_mode`**: Mode for hypothesis generation ("generate", "direct", "direct hypothesis", "csv_batch")
- **`num_keywords`**: Number of keywords to sample
- **`num_sampling_runs`**: Number of sampling runs to perform
- **`csv_file`**: Path to CSV file for batch processing mode
- **`enable_planning_agent`**: Enable/disable the planning agent
- **`enable_code_execution`**: Enable/disable code execution

See `config.yml` in the parent directory for a complete list of configuration options.

## Key Features

1. **Modular Architecture**: Clean separation of concerns with logical module organization
2. **AI Agent System**: Multiple specialized agents (Scientist, Critic, Planner) working together
3. **Data Processing**: Comprehensive data preprocessing and caching
4. **Code Execution**: Safe execution of LLM-generated Python code
5. **Experiment Management**: Full experiment lifecycle management
6. **User Interface**: Rich console interface with colors and progress tracking
7. **Error Handling**: Robust error handling and recovery mechanisms
8. **Configuration**: Flexible configuration system

## Dependencies

- Python 3.8+
- HuggingFace Transformers
- PyTorch
- vLLM (optional, for vLLM backend)
- Pandas
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Seaborn
- PyYAML
- tqdm

See the main README for installation instructions.

## Development

The package is designed for easy extension and modification:

1. **Adding New Agents**: Extend the base `Agent` class in `agents/models.py`
2. **Custom Data Processing**: Add new processors in the `data/` module
3. **New Experiment Types**: Extend `ExperimentRunner` in `experiment/runner.py`
4. **UI Enhancements**: Modify components in the `ui/` module

## Migration from Old Structure

The package has been reorganized from a flat structure to a modular one. Key changes:

- `discovery_agent.py` → `core/agent.py`
- `models.py` → `agents/models.py`
- `experiment_runner.py` → `experiment/runner.py`
- `data_processing.py` → `data/processor.py`
- `ui.py` → `ui/interface.py`
- `utils.py` → `core/utils.py` and `execution/utils.py`

All functionality remains the same, but imports need to be updated to use the new module structure.
