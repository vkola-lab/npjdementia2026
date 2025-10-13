"""
AI agents and models for the Discovery Agent.

This module contains all AI-related components including:
- Model client for LLM interactions
- Base agent class
- Specialized agents (Scientist, Critic, Planner)
- Hypothesis generator
"""

from .models import ModelClient, Agent, HypothesisGenerator
from .scientist import create_scientist
from .critic import create_critic
from .planner import create_enhanced_planning_agent

__all__ = [
    'ModelClient',
    'Agent', 
    'HypothesisGenerator',
    'create_scientist',
    'create_critic',
    'create_enhanced_planning_agent'
]
