"""
File management module for the Discovery Agent.

This module handles all file-related operations including:
- Working directory management
- CSV tracking and logging
- File cleanup and organization
- Filename generation with configuration
"""

import os
import csv
import shutil
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from ..ui.interface import ui
from ..core.config import config as default_config


class FileManager:
    """Handles all file management operations for the Discovery Agent."""
    
    def __init__(self, config=None):
        """Initialize the file manager with configuration."""
        self.config = config if config is not None else default_config
        self.csv_file_path = None
    
    def generate_filename_with_config(self, base_name: str, extension: str = "", include_timestamp: bool = True) -> str:
        """
        Generate a filename that includes important config variables.
        
        Args:
            base_name (str): Base name for the file
            extension (str): File extension (e.g., '.csv', '.json')
            include_timestamp (bool): Whether to include timestamp in filename
        
        Returns:
            str: Generated filename with config variables
        """
        # Get important config variables
        config_parts = []
        
        # Model name (shortened)
        model_short = self.config.model_name.split('/')[-1].replace('-', '_') if hasattr(self.config, 'model_name') else 'unknown'
        config_parts.append(f"model_{model_short}")
        
        # Hypothesis mode
        if hasattr(self.config, 'hypothesis_mode'):
            config_parts.append(f"mode_{self.config.hypothesis_mode}")
        
        # Number of keywords
        if hasattr(self.config, 'num_keywords'):
            config_parts.append(f"kw_{self.config.num_keywords}")
        
        # Number of sampling runs
        if hasattr(self.config, 'num_sampling_runs'):
            config_parts.append(f"runs_{self.config.num_sampling_runs}")
        
        # Outer loop iterations
        if hasattr(self.config, 'outer_loop_iterations'):
            config_parts.append(f"outer_{self.config.outer_loop_iterations}")
        
        # Auto approve after attempts
        if hasattr(self.config, 'auto_approve_after_attempts'):
            config_parts.append(f"auto_{self.config.auto_approve_after_attempts}")
        
        # Join config parts
        config_str = "_".join(config_parts)
        
        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{config_str}_{timestamp}{extension}"
        else:
            filename = f"{base_name}_{config_str}{extension}"
        
        return filename
    
    def cleanup_working_directory(self, recreate: bool = True) -> None:
        """
        Clean up the working directory and optionally recreate it.
        
        Args:
            recreate (bool): Whether to recreate the directory after cleanup
        """
        working_dir = Path("working")
        
        if recreate:
            # Delete and recreate the entire directory (for initialization)
            if working_dir.exists():
                try:
                    shutil.rmtree(working_dir)
                    ui.print_info(f"🗑️ Deleted existing working directory")
                except Exception as e:
                    ui.print_warning(f"⚠️ Could not delete working directory: {str(e)}")
            
            # Recreate working directory
            try:
                working_dir.mkdir(exist_ok=True)
                ui.print_info(f"📁 Created fresh working directory")
            except Exception as e:
                ui.print_error(f"Could not create working directory: {str(e)}", "working_directory_creation")
                raise
        else:
            # Just clean the contents (for between hypotheses)
            if working_dir.exists():
                try:
                    # Remove all files and subdirectories in the working directory
                    for item in working_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    ui.print_info(f"✅ Cleaned up working directory")
                except Exception as e:
                    ui.print_warning(f"⚠️ Could not fully clean working directory: {str(e)}")
            else:
                ui.print_info(f"📁 Working directory does not exist, no cleanup needed")
    
    def initialize_csv_tracking(self) -> None:
        """Initialize CSV file for tracking individual hypothesis approval status and details."""
        # Use analysis/results directory (created in config.py)
        results_dir = Path("analysis/results")
        
        # Generate filename with config variables
        csv_filename = self.generate_filename_with_config("hypothesis_approval_tracking", ".csv")
        self.csv_file_path = results_dir / csv_filename
        
        # Create new CSV file with headers
        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'run_number', 'keywords', 'hypothesis_index', 'hypothesis_text', 
                'hypothesis_approved', 'outer_iterations_used', 'inner_iterations_used', 
                'turns_to_approve', 'max_outer_iterations', 'max_inner_iterations', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        ui.print_info(f"📊 Created new CSV tracking file: {self.csv_file_path}")
    
    def write_csv_entry(self, run_number: int, keywords: list, hypothesis_index: int, 
                       hypothesis_text: str, outer_iterations_used: int, inner_iterations_used: int, 
                       hypothesis_approved: bool, turns_to_approve: Optional[int] = None) -> None:
        """
        Write an entry to the CSV tracking file for individual hypothesis.
        
        Args:
            run_number (int): The run number
            keywords (list): List of keywords used
            hypothesis_index (int): Index of the hypothesis
            hypothesis_text (str): The hypothesis text
            outer_iterations_used (int): Number of outer iterations used
            inner_iterations_used (int): Number of inner iterations used
            hypothesis_approved (bool): Whether the hypothesis was approved
            turns_to_approve (Optional[int]): Number of turns to approval (if approved)
        """
        # Check if CSV file path is initialized
        if self.csv_file_path is None:
            ui.print_warning("⚠️ CSV tracking file not initialized. Cannot write entry.")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        keywords_str = ', '.join(keywords)
        
        with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'run_number', 'keywords', 'hypothesis_index', 'hypothesis_text', 
                'hypothesis_approved', 'outer_iterations_used', 'inner_iterations_used', 
                'turns_to_approve', 'max_outer_iterations', 'max_inner_iterations', 'timestamp'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'run_number': run_number,
                'keywords': keywords_str,
                'hypothesis_index': hypothesis_index,
                'hypothesis_text': hypothesis_text,
                'hypothesis_approved': hypothesis_approved,
                'outer_iterations_used': outer_iterations_used,
                'inner_iterations_used': inner_iterations_used,
                'turns_to_approve': turns_to_approve if hypothesis_approved else None,
                'max_outer_iterations': self.config.outer_loop_iterations,
                'max_inner_iterations': self.config.auto_approve_after_attempts,
                'timestamp': timestamp
            })
        
        # Create informative log message
        if hypothesis_approved and turns_to_approve is not None:
            log_msg = f"Approved after {outer_iterations_used} outer iterations, {inner_iterations_used} turns in final iteration"
        elif not hypothesis_approved:
            log_msg = f"Not approved after {outer_iterations_used} outer iterations, {inner_iterations_used} turns in final iteration"
        else:
            log_msg = "Not Approved"
        
        ui.print_info(f"💾 Recorded hypothesis {hypothesis_index} data to CSV: {log_msg}")
    
    def get_csv_file_path(self) -> Optional[Path]:
        """Get the current CSV file path."""
        return self.csv_file_path
    
    def create_results_directory(self) -> None:
        """Create the results directory if it doesn't exist."""
        results_dir = Path("analysis/results")
        results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_responses_directory(self) -> None:
        """Create the responses directory if it doesn't exist."""
        responses_dir = Path("analysis/responses")
        responses_dir.mkdir(parents=True, exist_ok=True)
    
    def ensure_directories_exist(self) -> None:
        """Ensure all required directories exist."""
        self.create_results_directory()
        self.create_responses_directory()
        # Note: working directory is created/deleted dynamically
    
    def cleanup_old_files(self, directory: str, pattern: str = "*", max_age_days: int = 30) -> None:
        """
        Clean up old files in a directory.
        
        Args:
            directory (str): Directory to clean up
            pattern (str): File pattern to match
            max_age_days (int): Maximum age of files in days
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return
            
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
            files_removed = 0
            
            for file_path in directory_path.glob(pattern):
                if file_path.is_file():
                    file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        files_removed += 1
            
            if files_removed > 0:
                ui.print_info(f"🧹 Cleaned up {files_removed} old files from {directory}")
                
        except Exception as e:
            ui.print_warning(f"⚠️ Could not clean up old files in {directory}: {str(e)}")
    
    def get_file_size_mb(self, file_path: str) -> float:
        """
        Get file size in MB.
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            float: File size in MB
        """
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0
    
    def get_directory_size_mb(self, directory: str) -> float:
        """
        Get total size of directory in MB.
        
        Args:
            directory (str): Path to the directory
        
        Returns:
            float: Total size in MB
        """
        try:
            total_size = 0
            directory_path = Path(directory)
            if directory_path.exists():
                for file_path in directory_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0
