"""
Discovery Agent Package

A comprehensive AI-powered scientific discovery system that uses
Scientist and Critic agents to analyze data
and generate insights through iterative conversation.

Main Components:
- DiscoveryAgent: Main orchestrator class
- ModelClient: HuggingFace model interface
- ConversationManager: Manages conversation flow
- CodeExecutor: Executes Python code safely
- UI: User interface components
- Logger: Logging and monitoring
"""

from .discovery_agent import DiscoveryAgentSystem, main_discovery_agent
from .models import ModelClient, create_scientist, create_critic
from .conversation import ConversationManager
from .code_executor import CodeExecutor
from .ui import ui
from .config import config

__version__ = "1.0.0"
__author__ = "Discovery Agent Team"

__all__ = [
    "DiscoveryAgentSystem",
    "main_discovery_agent",
    "ModelClient", 
    "create_scientist",
    "create_critic",
    "ConversationManager",
    "CodeExecutor",
    "ui",
    "config"
]
