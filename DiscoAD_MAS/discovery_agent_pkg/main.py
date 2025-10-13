"""
Main entry point for the Discovery Agent Package.

This module provides the main entry point for running the Discovery Agent system.
It handles initialization, configuration, and execution of the discovery process.
"""

import os
import sys
import argparse
from typing import Optional

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.config import Config
from core.agent import DiscoveryAgentSystem
from ui.interface import ui
from core.utils import cleanup_resources


def main():
    """Main entry point for the Discovery Agent."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Discovery Agent - AI-powered scientific discovery system")
        parser.add_argument("--config", type=str, help="Path to configuration file")
        parser.add_argument("--dataset-folder", type=str, help="Path to dataset folder")
        parser.add_argument("--model-name", type=str, help="Model name to use")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument("--hypothesis", type=str, help="Direct hypothesis to test")
        parser.add_argument("--hypothesis-mode", type=str, choices=["generate", "direct", "direct hypothesis", "csv_batch"], 
                          default="generate", help="Hypothesis generation mode")
        parser.add_argument("--csv-file", type=str, help="CSV file path containing hypotheses when hypothesis-mode is csv_batch")
        
        args = parser.parse_args()
        
        # Create configuration
        config = Config()
        
        # Override config with command line arguments if provided
        if args.dataset_folder:
            config.dataset_folder = args.dataset_folder
        if args.model_name:
            config.model_name = args.model_name
        if args.verbose:
            config.verbose = True
        if args.hypothesis:
            config.hypothesis = args.hypothesis
        if args.hypothesis_mode:
            config.hypothesis_mode = args.hypothesis_mode
        if args.csv_file:
            config.csv_file = args.csv_file
        
        # Print banner
        ui.print_info("🔬 Discovery Agent - AI-Powered Scientific Discovery System")
        ui.print_separator()
        
        # Initialize and run agent
        ui.print_info("🔧 Initializing Discovery Agent System...")
        agent = DiscoveryAgentSystem(config)
        ui.print_info("🚀 Starting Discovery Agent...")
        agent.run()
        
    except KeyboardInterrupt:
        ui.print_info("🛑 Interrupted by user.")
        cleanup_resources()
    except ImportError as e:
        ui.print_error(f"Import error: {str(e)}", "main")
        ui.print_info("💡 Check if all required packages are installed.")
        cleanup_resources()
        sys.exit(1)
    except OSError as e:
        ui.print_error(f"System error: {str(e)}", "main")
        ui.print_info("💡 Check GPU availability and CUDA setup.")
        cleanup_resources()
        sys.exit(1)
    except RuntimeError as e:
        ui.print_error(f"Runtime error: {str(e)}", "main")
        ui.print_info("💡 This might be a model loading or GPU memory issue.")
        cleanup_resources()
        sys.exit(1)
    except Exception as e:
        ui.print_error(f"Unexpected error: {str(e)}", "main")
        ui.print_info("💡 Check the error details above for troubleshooting.")
        import traceback
        ui.print_error(f"Full traceback: {traceback.format_exc()}", "debug")
        cleanup_resources()
        sys.exit(1)
    finally:
        ui.print_info("👋 Discovery Agent session ended.")


if __name__ == "__main__":
    main()
