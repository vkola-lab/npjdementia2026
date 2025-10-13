"""
Experiment management for the Discovery Agent.

This module contains all experiment-related components including:
- Experiment runner for orchestrating experiments
- Hypothesis management for hypothesis generation and tracking
- Conversation management for handling agent conversations
"""

from .runner import ExperimentRunner
from .hypothesis import HypothesisManager
from .conversation import ConversationManager

__all__ = [
    'ExperimentRunner',
    'HypothesisManager',
    'ConversationManager'
]
