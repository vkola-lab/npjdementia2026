"""
Main Discovery Agent class.

This module contains the core DiscoveryAgent class that orchestrates
the entire discovery process with proper error handling and state management.
"""

import signal
import sys
from typing import Optional

# Optional emoji import with fallback
try:
    import emoji
    def emojize(text):
        return emoji.emojize(text)
except ImportError:
    def emojize(text):
        return text  # Return text without emoji processing

from .config import config as default_config
from ..agents.models import ModelClient
from ..ui.interface import ui
from ..experiment.runner import ExperimentRunner
from ..data.file_manager import FileManager

# Import cleanup functions from utils
from .utils import cleanup_gpu_processes, signal_handler, sigtstp_handler, cleanup_resources


class DiscoveryAgentSystem:
    """Main Discovery Agent class."""
    
    def __init__(self, config=None):
        """Initialize the Discovery Agent with all components."""
        # Use provided config or fall back to default
        self.config = config if config is not None else default_config
        
        try:
            ui.print_info(f"{emojize('⚙️')} Initializing Discovery Agent...")
            
            # Initialize model client
            self.model_client = ModelClient(self.config)
            
            # Initialize file manager and setup directories
            self.file_manager = FileManager(self.config)
            self.file_manager.ensure_directories_exist()
            self.file_manager.cleanup_working_directory(recreate=True)
            self.file_manager.initialize_csv_tracking()
            
            # Initialize experiment runner with all components
            self.experiment_runner = ExperimentRunner(self.model_client, self.config, self.file_manager)
            
            # Initialize data cache if dataset folder is provided
            if hasattr(self.config, 'dataset_folder') and self.config.dataset_folder:
                ui.print_info(f"📊 Initializing data cache from dataset folder: {self.config.dataset_folder}")
                if not self.experiment_runner.initialize_data_cache(self.config.dataset_folder):
                    ui.print_warning("⚠️ Failed to initialize data cache. Some features may not work properly.")
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGTSTP, sigtstp_handler)
            
            ui.print_success(f"{emojize('🎉')} Discovery Agent initialized successfully!")
        
        except Exception as e:
            ui.print_error(str(e), "initialization")
            raise
    
    def run(self):
        """Run the main discovery agent loop with keyword sampling."""
        try:
            # Initialize sampling run
            init_result = self.experiment_runner.initialize_sampling_run()
            if init_result[0] is None and init_result[1] is None:  # Both keywords and num_runs are None (error case)
                return
            
            keywords, num_runs, verbose, total_hypotheses, total_approved = init_result
            
            # Run the main sampling iterations
            total_hypotheses, total_approved = self.experiment_runner.run_sampling_iterations(
                keywords, num_runs, verbose, total_hypotheses, total_approved
            )
            
            # Print final experiment summary
            self.experiment_runner.print_experiment_summary(num_runs, total_hypotheses, total_approved, verbose)
            
        except KeyboardInterrupt:
            ui.print_info(f"{emojize('🛑')} Interrupted by user. Cleaning up...")
            cleanup_resources()
        except Exception as e:
            ui.print_error(str(e), "main loop")
            cleanup_resources()
            raise


def main_discovery_agent(config=None):
    """Main entry point for the Discovery Agent."""
    try:
        # Print banner
        ui.print_banner()
        
        # Initialize and run agent with better error handling
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
        ui.close_log()