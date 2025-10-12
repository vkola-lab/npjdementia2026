#!/usr/bin/env python3
"""
Discovery Agent - Main Entry Point

This is the main entry point for the Discovery Agent system.
It provides a clean interface to the modular discovery agent components.
"""
from dotenv import load_dotenv
load_dotenv()


from discovery_agent_pkg.config import Config
from discovery_agent_pkg import main_discovery_agent

if __name__ == "__main__":
    # Initialize config with command line arguments
    config = Config()
    
    main_discovery_agent(config)