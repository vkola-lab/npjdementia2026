"""
Simple configuration for the Discovery Agent.
Gets configuration from command line arguments.
"""

import argparse
from pathlib import Path

class Config:
    def __init__(self, args=None):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='DiscoAD_MAS Discovery Agent')
        parser.add_argument('--model-name', default='Qwen/Qwen3-4B-Instruct-2507', help='Model name')
        parser.add_argument('--device', default='auto', help='Device (auto, cuda, cpu)')
        parser.add_argument('--max-tokens', type=int, default=8192, help='Max new tokens')
        parser.add_argument('--max-conversation-length', type=int, default=6, help='Max conversation length')
        parser.add_argument('--auto-approve-after', type=int, default=20, help='Auto approve after N attempts')
        parser.add_argument('--code-timeout', type=int, default=30, help='Code execution timeout in seconds')
        
        # User input configuration parameters
        parser.add_argument('--hypothesis-mode', choices=['generate', 'direct', 'direct hypothesis'], default='generate', 
                          help='Whether to generate multiple hypotheses (generate), use keywords directly (direct), or use provided hypothesis (direct hypothesis)')
        parser.add_argument('--hypothesis', type=str, help='Direct hypothesis to use when hypothesis-mode is "direct hypothesis"')
        parser.add_argument('--dataset-folder', type=str, help='Path to folder containing CSV and JSON files')
        parser.add_argument('--keywords-file', type=str, default='adrd_keywords.txt', 
                          help='Path to keywords file for sampling (default: adrd_keywords.txt)')
        parser.add_argument('--num-keywords', type=int, default=3, help='Number of keywords to sample per run (default: 3)')
        parser.add_argument('--num-sampling-runs', type=int, default=10, help='Number of keyword sampling runs to perform')
        parser.add_argument('--outer-loop-iterations', type=int, default=3, help='Number of outer loop iterations (k) for data prep -> planner -> scientist/critic cycle')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output (default: quiet mode)')
        
        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(args)
        
        # Model settings
        self.model_name = args.model_name
        self.device = args.device
        self.max_new_tokens = args.max_tokens
        
        # Agent settings
        self.max_conversation_length = args.max_conversation_length
        self.enable_code_execution = True  # Always enable code execution
        self.auto_approve_after_attempts = args.auto_approve_after
        self.code_timeout = args.code_timeout
        
        # User input settings
        self.hypothesis_mode = args.hypothesis_mode
        self.hypothesis = args.hypothesis
        self.dataset_folder = args.dataset_folder
        self.keywords_file = args.keywords_file
        self.num_keywords = args.num_keywords
        self.num_sampling_runs = args.num_sampling_runs
        self.outer_loop_iterations = args.outer_loop_iterations
        self.verbose = args.verbose
        
        # Create directories
        Path("responses").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        # Note: working directory is now created/deleted in DiscoveryAgentSystem.__init__
        # Path("working").mkdir(exist_ok=True)
        
        # Print configuration info
        print(f"🔧 Configuration loaded:")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Max tokens: {self.max_new_tokens}")
        print(f"   Code timeout: {self.code_timeout}s")
        print(f"   Hypothesis mode: {self.hypothesis_mode}")
        if self.hypothesis_mode == "direct hypothesis" and self.hypothesis:
            print(f"   Direct hypothesis: {self.hypothesis}")
        print(f"   Keywords file: {self.keywords_file}")
        print(f"   Number of keywords per run: {self.num_keywords}")
        print(f"   Number of sampling runs: {self.num_sampling_runs}")
        print(f"   Outer loop iterations: {self.outer_loop_iterations}")
        print(f"   Verbose output: {self.verbose}")
        if self.dataset_folder:
            print(f"   Dataset folder: {self.dataset_folder}")

# Global configuration instance (will be set by main script)
config = None
