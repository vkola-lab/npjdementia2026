#!/usr/bin/env python3
"""
Discovery Agent Entry Point

This is the main entry point for the Discovery Agent system.
It imports and runs the main discovery agent functionality.
"""

import sys
import os
import random
import numpy as np
import torch
# Add the discovery_agent_pkg to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'discovery_agent_pkg'))

# Import and run the main function
from discovery_agent_pkg import main_discovery_agent
from discovery_agent_pkg import Config


def set_seed(seed: int):
    """Set the seed for the random number generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Create configuration (handles all CLI args internally)
    config = Config()
    set_seed(config.seed)
    # Run the main discovery agent
    main_discovery_agent(config)
