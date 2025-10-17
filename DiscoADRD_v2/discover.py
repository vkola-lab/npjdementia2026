#!/usr/bin/env python3
"""
Discovery Agent Entry Point

This is the main entry point for the Discovery Agent system.
It imports and runs the main discovery agent functionality.
"""

import sys
import os

# Add the discovery_agent_pkg to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'discovery_agent_pkg'))

# Import and run the main function
from discovery_agent_pkg import main_discovery_agent
from discovery_agent_pkg import Config

if __name__ == "__main__":
    # Create configuration (handles all CLI args internally)
    config = Config()
    
    # Run the main discovery agent
    main_discovery_agent(config)
