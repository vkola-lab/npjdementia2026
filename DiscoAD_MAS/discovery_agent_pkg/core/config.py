"""
Configuration for the Discovery Agent.
Loads configuration from YAML file with command line overrides.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    def __init__(self, args=None, config_file: str = "config.yml"):
        """
        Initialize configuration from YAML file with command line overrides.
        
        Args:
            args: Command line arguments (optional)
            config_file: Path to YAML configuration file
        """
        # Parse command line arguments first to get config file path
        parser = argparse.ArgumentParser(description='DiscoAD_MAS Discovery Agent')
        parser.add_argument('--config', type=str, default=config_file, help='Path to YAML config file')
        
        # Parse just the config argument first
        if args is None:
            parsed_args, remaining_args = parser.parse_known_args()
        else:
            parsed_args, remaining_args = parser.parse_known_args(args)
        
        # Load YAML configuration using the specified config file
        self.yaml_config = self._load_yaml_config(parsed_args.config)
        
        # Now parse remaining arguments
        parser = argparse.ArgumentParser(description='DiscoAD_MAS Discovery Agent')
        parser.add_argument('--hypothesis-mode', choices=['generate', 'direct', 'direct hypothesis', 'csv_batch'], 
                          help='Override hypothesis mode from config')
        parser.add_argument('--hypothesis', type=str, help='Direct hypothesis to use when hypothesis-mode is "direct hypothesis"')
        parser.add_argument('--csv-file', type=str, help='CSV file path containing hypotheses when hypothesis-mode is "csv_batch"')
        parser.add_argument('--start-index', type=int, help='Start processing CSV from this index (1-based) when hypothesis-mode is "csv_batch"')
        parser.add_argument('--dataset-folder', type=str, help='Override dataset folder from config')
        parser.add_argument('--verbose', action='store_true', help='Override verbose setting from config')
        parser.add_argument('--disable-planning-agent', action='store_true', help='Disable the planning agent (overrides config)')
        
        # Parse remaining arguments
        args = parser.parse_args(remaining_args)
        
        # Apply command line overrides to YAML config
        self._apply_cli_overrides(args)
        
        # Set attributes from merged configuration
        self._set_attributes()
        
        # Create directories
        self._create_directories()
        
        # Print configuration info
        self._print_config_info()
    
    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"📄 Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️ Config file {config_file} not found, using defaults")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if YAML file is not available."""
        return {
            'model': {
                'name': 'Qwen/Qwen3-4B-Instruct-2507',
                'device': 'auto',
                'max_tokens': 8192
            },
            'agents': {
                'max_conversation_length': 6,
                'auto_approve_after_attempts': 20,
                'code_timeout': 30,
                'enable_planning_agent': True,
                'enable_code_execution': True
            },
            'experiment': {
                'num_keywords': 3,
                'num_sampling_runs': 10,
                'outer_loop_iterations': 3
            },
            'paths': {
                'keywords_file': 'adrd_keywords.txt',
                'analysis_dir': 'analysis',
                'responses_dir': 'analysis/responses',
                'results_dir': 'analysis/results',
                'qsub_runs_dir': 'analysis/qsub_runs',
                'working_dir': 'working'
            },
            'output': {
                'verbose': False,
                'save_responses': True,
                'save_results': True
            }
        }
    
    def _apply_cli_overrides(self, args) -> None:
        """Apply command line argument overrides to YAML config."""
        # Override hypothesis mode if provided
        if args.hypothesis_mode:
            if 'experiment' not in self.yaml_config:
                self.yaml_config['experiment'] = {}
            self.yaml_config['experiment']['hypothesis_mode'] = args.hypothesis_mode
        
        # Override hypothesis if provided
        if args.hypothesis:
            if 'experiment' not in self.yaml_config:
                self.yaml_config['experiment'] = {}
            self.yaml_config['experiment']['hypothesis'] = args.hypothesis
        
        # Override CSV file if provided
        if args.csv_file:
            if 'experiment' not in self.yaml_config:
                self.yaml_config['experiment'] = {}
            self.yaml_config['experiment']['csv_file'] = args.csv_file
        
        # Override start index if provided
        if args.start_index:
            if 'experiment' not in self.yaml_config:
                self.yaml_config['experiment'] = {}
            self.yaml_config['experiment']['start_index'] = args.start_index
        
        # Override dataset folder if provided
        if args.dataset_folder:
            if 'paths' not in self.yaml_config:
                self.yaml_config['paths'] = {}
            self.yaml_config['paths']['dataset_folder'] = args.dataset_folder
        
        # Override verbose if provided
        if args.verbose:
            if 'output' not in self.yaml_config:
                self.yaml_config['output'] = {}
            self.yaml_config['output']['verbose'] = True
        
        # Override planning agent if disabled
        if args.disable_planning_agent:
            if 'agents' not in self.yaml_config:
                self.yaml_config['agents'] = {}
            self.yaml_config['agents']['enable_planning_agent'] = False
    
    def _set_attributes(self) -> None:
        """Set object attributes from merged configuration."""
        # Model settings
        model_config = self.yaml_config.get('model', {})
        self.model_name = model_config.get('name', 'Qwen/Qwen3-4B-Instruct-2507')
        self.device = model_config.get('device', 'auto')
        self.max_new_tokens = model_config.get('max_tokens', 8192)
        
        # Agent settings
        agents_config = self.yaml_config.get('agents', {})
        self.max_conversation_length = agents_config.get('max_conversation_length', 6)
        self.enable_code_execution = agents_config.get('enable_code_execution', True)
        self.auto_approve_after_attempts = agents_config.get('auto_approve_after_attempts', 20)
        self.code_timeout = agents_config.get('code_timeout', 30)
        self.enable_planning_agent = agents_config.get('enable_planning_agent', True)
        
        # Experiment settings
        experiment_config = self.yaml_config.get('experiment', {})
        self.hypothesis_mode = experiment_config.get('hypothesis_mode', 'generate')
        self.hypothesis = experiment_config.get('hypothesis', None)
        self.csv_file = experiment_config.get('csv_file', None)
        self.start_index = experiment_config.get('start_index', 1)  # Default to 1 (first row)
        self.num_keywords = experiment_config.get('num_keywords', 3)
        self.num_sampling_runs = experiment_config.get('num_sampling_runs', 10)
        self.outer_loop_iterations = experiment_config.get('outer_loop_iterations', 3)
        
        # Path settings
        paths_config = self.yaml_config.get('paths', {})
        self.keywords_file = paths_config.get('keywords_file', 'adrd_keywords.txt')
        self.dataset_folder = paths_config.get('dataset_folder', None)
        self.analysis_dir = paths_config.get('analysis_dir', 'analysis')
        self.responses_dir = paths_config.get('responses_dir', 'analysis/responses')
        self.results_dir = paths_config.get('results_dir', 'analysis/results')
        self.qsub_runs_dir = paths_config.get('qsub_runs_dir', 'analysis/qsub_runs')
        self.working_dir = paths_config.get('working_dir', 'working')
        
        # Output settings
        output_config = self.yaml_config.get('output', {})
        self.verbose = output_config.get('verbose', False)
        self.save_responses = output_config.get('save_responses', True)
        self.save_results = output_config.get('save_results', True)
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        Path(self.responses_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.qsub_runs_dir).mkdir(parents=True, exist_ok=True)
        # Note: working directory is now created/deleted in DiscoveryAgentSystem.__init__
    
    def _print_config_info(self) -> None:
        """Print configuration information."""
        print(f"🔧 Configuration loaded:")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Max tokens: {self.max_new_tokens}")
        print(f"   Code timeout: {self.code_timeout}s")
        print(f"   Planning agent: {'enabled' if self.enable_planning_agent else 'disabled'}")
        print(f"   Hypothesis mode: {self.hypothesis_mode}")
        if self.hypothesis_mode == "direct hypothesis" and self.hypothesis:
            print(f"   Direct hypothesis: {self.hypothesis}")
        if self.hypothesis_mode == "csv_batch" and self.start_index > 1:
            print(f"   Start index: {self.start_index}")
        print(f"   Keywords file: {self.keywords_file}")
        print(f"   Number of keywords per run: {self.num_keywords}")
        print(f"   Number of sampling runs: {self.num_sampling_runs}")
        print(f"   Outer loop iterations: {self.outer_loop_iterations}")
        print(f"   Verbose output: {self.verbose}")
        if self.dataset_folder:
            print(f"   Dataset folder: {self.dataset_folder}")

# Global configuration instance (will be set by main script)
config = None
