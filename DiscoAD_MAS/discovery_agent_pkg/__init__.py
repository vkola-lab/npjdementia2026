"""
Discovery Agent Package

A comprehensive AI-powered scientific discovery system that uses multiple agents
(Scientist, Critic, Planning Agent) to test hypotheses and analyze data.

Main Components:
- Core: Main agent system and configuration
- Agents: AI model clients and agent implementations
- Data: Data processing, caching, and file management
- Experiment: Experiment execution and hypothesis management
- Execution: Code execution utilities
- UI: User interface components

Usage:
    from discovery_agent_pkg import DiscoveryAgentSystem
    
    # Initialize and run the discovery agent
    agent = DiscoveryAgentSystem()
    agent.run()
"""

# Import only the config and ui modules that don't have heavy dependencies
# Import Config directly to avoid triggering the core module imports
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Import config module directly
import importlib.util
spec = importlib.util.spec_from_file_location("config", os.path.join(os.path.dirname(__file__), "core", "config.py"))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
Config = config_module.Config

# Import ui module directly
spec = importlib.util.spec_from_file_location("ui", os.path.join(os.path.dirname(__file__), "ui", "interface.py"))
ui_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui_module)
ui = ui_module.ui

# Lazy imports for heavy dependencies
def _get_discovery_agent_system():
    try:
        from .core.agent import DiscoveryAgentSystem
        return DiscoveryAgentSystem
    except ImportError as e:
        if "torch" in str(e):
            raise ImportError("PyTorch is required but not installed. Please install torch to use DiscoveryAgentSystem.")
        raise

def _get_main_discovery_agent():
    try:
        from .core.agent import main_discovery_agent
        return main_discovery_agent
    except ImportError as e:
        if "torch" in str(e):
            raise ImportError("PyTorch is required but not installed. Please install torch to use main_discovery_agent.")
        raise

# Expose the lazy imports at module level
def __getattr__(name):
    if name == 'DiscoveryAgentSystem':
        return _get_discovery_agent_system()
    elif name == 'main_discovery_agent':
        return _get_main_discovery_agent()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "1.0.0"
__author__ = "Discovery Agent Team"

__all__ = [
    'DiscoveryAgentSystem',
    'main_discovery_agent',
    'Config', 
    'ui'
]

# Make the lazy imports available at module level
# These need to be callable functions that return the actual classes/functions
DiscoveryAgentSystem = lambda *args, **kwargs: _get_discovery_agent_system()(*args, **kwargs)
main_discovery_agent = lambda *args, **kwargs: _get_main_discovery_agent()(*args, **kwargs)