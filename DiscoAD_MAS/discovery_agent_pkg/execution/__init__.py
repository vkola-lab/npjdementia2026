"""
Execution module for the Discovery Agent.

This module handles code execution and related utilities for running
LLM-generated Python code in a controlled environment.
"""

from .executor import CodeExecutor
from .utils import execute_code_from_response

__all__ = ['CodeExecutor', 'execute_code_from_response']
