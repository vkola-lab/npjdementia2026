"""
Discovery Agent Package

A comprehensive AI-powered scientific discovery system that uses
Scientist and Critic agents to analyze data
and generate insights through iterative conversation.

Main Components:
- DiscoveryAgentSystem: Main orchestrator class
- ModelClient: HuggingFace model interface
- ConversationManager: Manages conversation flow
- CodeExecutor: Executes Python code safely
- UI: User interface components
- DataProcessor: Handles dataset operations and preprocessing
- HypothesisManager: Manages hypothesis generation and tracking
- ExperimentRunner: Orchestrates experiment execution
- FileManager: Handles file operations and tracking
"""

from .discovery_agent import DiscoveryAgentSystem, main_discovery_agent
from .models import ModelClient, create_scientist, create_critic, HypothesisGenerator
from .conversation import ConversationManager
from .code_executor import CodeExecutor
from .ui import ui
from .config import config
from .data_processing import DataProcessor
from .hypothesis_management import HypothesisManager
from .experiment_runner import ExperimentRunner
from .file_management import FileManager

__version__ = "1.0.0"
__author__ = "Discovery Agent Team"

__all__ = [
    "DiscoveryAgentSystem",
    "main_discovery_agent",
    "ModelClient", 
    "create_scientist",
    "create_critic",
    "HypothesisGenerator",
    "ConversationManager",
    "CodeExecutor",
    "ui",
    "config",
    "DataProcessor",
    "HypothesisManager",
    "ExperimentRunner",
    "FileManager"
]
