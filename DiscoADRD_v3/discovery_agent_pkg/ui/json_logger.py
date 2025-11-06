"""
JSON-based structured logging system for the Discovery Agent.

This module provides structured JSON logging that tracks:
- Experiment metadata and configuration
- Hypothesis processing with turns and iterations
- Agent responses and code execution results
- Final outcomes and timing information
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Try to import dataclasses, fall back to manual implementation if not available
try:
    from dataclasses import dataclass, asdict, field
    DATACLASSES_AVAILABLE = True
except ImportError:
    DATACLASSES_AVAILABLE = False
    
    # Fallback implementation
    def dataclass(cls):
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        cls.__init__ = __init__
        return cls
    
    def field(default_factory=None):
        return default_factory
    
    def asdict(obj):
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = asdict(value)
                elif isinstance(value, list):
                    result[key] = [asdict(item) if hasattr(item, '__dict__') else item for item in value]
                else:
                    result[key] = value
            return result
        return obj


@dataclass
class CodeExecutionResult:
    """Structured code execution result."""
    success: bool
    output: str = ""
    error: str = ""
    traceback: str = ""
    execution_time_seconds: Optional[float] = None


@dataclass
class Turn:
    """A single turn in the conversation."""
    turn: int
    timestamp: str
    agent: str
    response: str
    code_execution: Optional[CodeExecutionResult] = None

@dataclass
class SystemLog:
    """System log entry."""
    timestamp: str
    level: str
    message: str
    category: Optional[str] = None

@dataclass
class DataPreprocessingStep:
    """Data preprocessing step."""
    step_number: int
    step_name: str
    description: str
    timestamp: str
    status: str
    details: Optional[str] = None


@dataclass
class OuterIteration:
    """An outer iteration for a hypothesis."""
    iteration: int
    turns: List[Turn]
    final_status: str
    turns_to_completion: int
    execution_time_seconds: Optional[float] = None


@dataclass
class Hypothesis:
    """A complete hypothesis processing record."""
    hypothesis_index: int
    hypothesis_text: str
    keywords: List[str]
    outer_iterations: List[OuterIteration]
    final_status: str
    total_outer_iterations: int
    total_turns: int
    execution_time_seconds: float
    data_preprocessing_steps: List[DataPreprocessingStep] = field(default_factory=list)


@dataclass
class ExperimentMetadata:
    """Experiment metadata and configuration."""
    timestamp: str
    config: Dict[str, Any]
    dataset_info: Dict[str, str]


@dataclass
class ExperimentSummary:
    """Summary of the entire experiment."""
    total_hypotheses: int
    verified_hypotheses: int
    not_verified_hypotheses: int
    not_applicable_hypotheses: int
    total_execution_time_seconds: float


@dataclass
class ExperimentLog:
    """Complete experiment log structure."""
    experiment_metadata: ExperimentMetadata
    hypotheses: List[Hypothesis]
    experiment_summary: ExperimentSummary
    system_logs: List[SystemLog] = field(default_factory=list)
    data_preprocessing_steps: List[DataPreprocessingStep] = field(default_factory=list)


class JSONLogger:
    """JSON-based structured logger for the Discovery Agent."""
    
    def __init__(self, config=None):
        """Initialize the JSON logger."""
        self.config = config
        # Initialize FileManager for filename generation (lazy import to avoid import errors)
        try:
            # Try relative import first (works when loaded as part of package)
            try:
                from ..data.file_manager import FileManager
            except (ImportError, ValueError):
                # Fallback to absolute import (works when loaded via importlib)
                from discovery_agent_pkg.data.file_manager import FileManager
            self.file_manager = FileManager(config) if config else None
        except ImportError:
            self.file_manager = None
        self.log_data = None
        self.current_hypothesis = None
        self.current_outer_iteration = None
        self.current_turn = None
        self.start_time = None
        # Store the generated filename for consistent use
        self.log_filename = None
        
        # Initialize log structure
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize the log structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate filename with config variables if FileManager is available
        if self.file_manager:
            self.log_filename = self.file_manager.generate_filename_with_config("session_log", ".json")
            # Extract timestamp from generated filename for consistency
            # The filename format is: session_log_{config_str}_{timestamp}.json
            # Timestamp is always in format YYYYMMDD_HHMMSS at the end before .json
            filename_no_ext = self.log_filename.replace(".json", "")
            # Find the last two parts that match timestamp format (YYYYMMDD_HHMMSS)
            parts = filename_no_ext.split("_")
            if len(parts) >= 2:
                # Check if last two parts form a valid timestamp
                last_part = parts[-1]
                second_last = parts[-2]
                if len(second_last) == 8 and second_last.isdigit() and len(last_part) == 6 and last_part.isdigit():
                    timestamp = f"{second_last}_{last_part}"
            # If timestamp extraction failed, keep the original timestamp
        else:
            # Fallback to simple timestamp if no config available
            self.log_filename = f"session_log_{timestamp}.json"
        
        # Extract config information
        config_dict = {}
        if self.config:
            config_dict = {
                "model_name": getattr(self.config, 'model_name', 'unknown'),
                "seed": getattr(self.config, 'seed', 42),
                "hypothesis_mode": getattr(self.config, 'hypothesis_mode', 'unknown'),
                "num_keywords": getattr(self.config, 'num_keywords', 3),
                "num_sampling_runs": getattr(self.config, 'num_sampling_runs', 10),
                "outer_loop_iterations": getattr(self.config, 'outer_loop_iterations', 2),
                "auto_approve_after_attempts": getattr(self.config, 'auto_approve_after_attempts', 3)
            }
        
        dataset_info = {}
        if self.config:
            dataset_info = {
                "csv_file": getattr(self.config, 'csv_file', ''),
                "dataset_folder": getattr(self.config, 'dataset_folder', '')
            }
        
        self.log_data = ExperimentLog(
            experiment_metadata=ExperimentMetadata(
                timestamp=timestamp,
                config=config_dict,
                dataset_info=dataset_info
            ),
            hypotheses=[],
            experiment_summary=ExperimentSummary(
                total_hypotheses=0,
                verified_hypotheses=0,
                not_verified_hypotheses=0,
                not_applicable_hypotheses=0,
                total_execution_time_seconds=0.0
            ),
            system_logs=[],
            data_preprocessing_steps=[]
        )
    
    def start_hypothesis(self, hypothesis_index: int, hypothesis_text: str, keywords: List[str]):
        """Start logging a new hypothesis."""
        # Load existing JSON data if this is not the first hypothesis
        if hypothesis_index > 0:
            self._load_existing_json_data()
        
        # Clear global preprocessing steps for this new hypothesis
        self.log_data.data_preprocessing_steps = []
        
        self.current_hypothesis = Hypothesis(
            hypothesis_index=hypothesis_index,
            hypothesis_text=hypothesis_text,
            keywords=keywords,
            outer_iterations=[],
            final_status="",
            total_outer_iterations=0,
            total_turns=0,
            execution_time_seconds=0.0,
            data_preprocessing_steps=[]
        )
        self.start_time = time.time()
    
    def start_outer_iteration(self, iteration: int):
        """Start a new outer iteration."""
        self.current_outer_iteration = OuterIteration(
            iteration=iteration,
            turns=[],
            final_status="",
            turns_to_completion=0,
            execution_time_seconds=0.0
        )
        self.current_turn = None
    
    def add_turn(self, turn: int, agent: str, response: str, 
                 code_execution: Optional[CodeExecutionResult] = None):
        """Add a turn to the current outer iteration."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Clean up the response text for better JSON formatting
        cleaned_response = self._clean_response_text(response)
        
        turn_data = Turn(
            turn=turn,
            timestamp=timestamp,
            agent=agent,
            response=cleaned_response,
            code_execution=code_execution
        )
        
        if self.current_outer_iteration:
            self.current_outer_iteration.turns.append(turn_data)
    
    def add_system_log(self, level: str, message: str, category: Optional[str] = None):
        """Add a system log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = SystemLog(
            timestamp=timestamp,
            level=level,
            message=message,
            category=category
        )
        self.log_data.system_logs.append(log_entry)
    
    def add_preprocessing_step(self, step_number: int, step_name: str, description: str, 
                             status: str, details: Optional[str] = None):
        """Add a data preprocessing step to the current hypothesis."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        step = DataPreprocessingStep(
            step_number=step_number,
            step_name=step_name,
            description=description,
            timestamp=timestamp,
            status=status,
            details=details
        )
        # Add to current hypothesis if available, otherwise to global list
        if self.current_hypothesis:
            self.current_hypothesis.data_preprocessing_steps.append(step)
        else:
            self.log_data.data_preprocessing_steps.append(step)
    
    def _clean_response_text(self, text: str) -> str:
        """Clean response text for better JSON formatting."""
        if not text:
            return text
        
        # Replace multiple consecutive newlines with single newline
        import re
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        # Replace multiple consecutive spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Replace any remaining problematic characters
        text = text.replace('\r', '')
        
        # Keep newlines as-is for proper multi-line formatting
        return text
    
    def end_outer_iteration(self, final_status: str):
        """End the current outer iteration."""
        if self.current_outer_iteration:
            self.current_outer_iteration.final_status = final_status
            self.current_outer_iteration.turns_to_completion = len(self.current_outer_iteration.turns)
            self.current_outer_iteration.execution_time_seconds = time.time() - self.start_time if self.start_time else 0.0
            
            if self.current_hypothesis:
                self.current_hypothesis.outer_iterations.append(self.current_outer_iteration)
    
    def end_hypothesis(self, final_status: str):
        """End the current hypothesis and write it to JSON immediately."""
        if self.current_hypothesis:
            self.current_hypothesis.final_status = final_status
            self.current_hypothesis.total_outer_iterations = len(self.current_hypothesis.outer_iterations)
            self.current_hypothesis.total_turns = sum(len(iter.turns) for iter in self.current_hypothesis.outer_iterations)
            self.current_hypothesis.execution_time_seconds = time.time() - self.start_time if self.start_time else 0.0
            
            # Add hypothesis to the log data
            self.log_data.hypotheses.append(self.current_hypothesis)
            
            # Update experiment summary
            self.log_data.experiment_summary.total_hypotheses += 1
            if final_status == "HYPOTHESIS VERIFIED":
                self.log_data.experiment_summary.verified_hypotheses += 1
            elif final_status == "NOT APPLICABLE":
                self.log_data.experiment_summary.not_applicable_hypotheses += 1
            else:
                self.log_data.experiment_summary.not_verified_hypotheses += 1
            
            self.log_data.experiment_summary.total_execution_time_seconds += self.current_hypothesis.execution_time_seconds
            
            # Write to JSON file immediately after each hypothesis
            self._write_hypothesis_to_json()
            
            # Clear current hypothesis to free memory
            self.current_hypothesis = None
            self.current_outer_iteration = None
    
    def _write_hypothesis_to_json(self):
        """Write the current log data to JSON file immediately."""
        # Create default file path
        responses_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "analysis", "responses")
        os.makedirs(responses_dir, exist_ok=True)
        # Use stored filename if available, otherwise fall back to timestamp-based name
        if self.log_filename:
            file_path = os.path.join(responses_dir, self.log_filename)
        else:
            timestamp = self.log_data.experiment_metadata.timestamp
            file_path = os.path.join(responses_dir, f"session_log_{timestamp}.json")
        
        # Convert dataclass to dict for JSON serialization
        log_dict = asdict(self.log_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    def _load_existing_json_data(self):
        """Load existing JSON data from file."""
        try:
            # Create default file path
            responses_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "analysis", "responses")
            # Use stored filename if available, otherwise fall back to timestamp-based name
            if self.log_filename:
                file_path = os.path.join(responses_dir, self.log_filename)
            else:
                timestamp = self.log_data.experiment_metadata.timestamp
                file_path = os.path.join(responses_dir, f"session_log_{timestamp}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Restore the log data from the file
                # Note: This is a simplified approach - in practice, you'd need to properly reconstruct
                # the dataclass objects from the JSON data
                self.log_data.hypotheses = existing_data.get('hypotheses', [])
                self.log_data.system_logs = existing_data.get('system_logs', [])
                self.log_data.data_preprocessing_steps = existing_data.get('data_preprocessing_steps', [])
                
                # Update experiment summary
                if 'experiment_summary' in existing_data:
                    summary = existing_data['experiment_summary']
                    self.log_data.experiment_summary.total_hypotheses = summary.get('total_hypotheses', 0)
                    self.log_data.experiment_summary.verified_hypotheses = summary.get('verified_hypotheses', 0)
                    self.log_data.experiment_summary.not_verified_hypotheses = summary.get('not_verified_hypotheses', 0)
                    self.log_data.experiment_summary.not_applicable_hypotheses = summary.get('not_applicable_hypotheses', 0)
                    self.log_data.experiment_summary.total_execution_time_seconds = summary.get('total_execution_time_seconds', 0.0)
        except Exception as e:
            # If loading fails, continue with current data
            print(f"Warning: Could not load existing JSON data: {e}")
    
    def save_log(self, file_path: Optional[str] = None) -> str:
        """Save the log to a JSON file."""
        if file_path is None:
            # Create default file path
            responses_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "analysis", "responses")
            os.makedirs(responses_dir, exist_ok=True)
            # Use stored filename if available, otherwise fall back to timestamp-based name
            if self.log_filename:
                file_path = os.path.join(responses_dir, self.log_filename)
            else:
                timestamp = self.log_data.experiment_metadata.timestamp
                file_path = os.path.join(responses_dir, f"session_log_{timestamp}.json")
        
        # Convert dataclass to dict for JSON serialization
        log_dict = asdict(self.log_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
        return file_path
    
    def save_log_realtime(self) -> str:
        """Save the log to a JSON file in real-time (updates existing file)."""
        # Create default file path
        responses_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "analysis", "responses")
        os.makedirs(responses_dir, exist_ok=True)
        # Use stored filename if available, otherwise fall back to timestamp-based name
        if self.log_filename:
            file_path = os.path.join(responses_dir, self.log_filename)
        else:
            timestamp = self.log_data.experiment_metadata.timestamp
            file_path = os.path.join(responses_dir, f"session_log_{timestamp}.json")
        
        # Convert dataclass to dict for JSON serialization
        log_dict = asdict(self.log_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
        return file_path
    
    def get_current_turn_count(self) -> int:
        """Get the current turn count for the current outer iteration."""
        if self.current_outer_iteration:
            return len(self.current_outer_iteration.turns)
        return 0
    
    def get_current_hypothesis_index(self) -> int:
        """Get the current hypothesis index."""
        if self.current_hypothesis:
            return self.current_hypothesis.hypothesis_index
        return -1


# Global JSON logger instance
json_logger = None


def get_json_logger() -> JSONLogger:
    """Get the global JSON logger instance."""
    global json_logger
    if json_logger is None:
        json_logger = JSONLogger()
    return json_logger


def initialize_json_logger(config=None):
    """Initialize the global JSON logger with config."""
    global json_logger
    json_logger = JSONLogger(config)


def save_json_log(file_path: Optional[str] = None) -> str:
    """Save the global JSON log."""
    global json_logger
    if json_logger is None:
        raise RuntimeError("JSON logger not initialized")
    return json_logger.save_log(file_path)
