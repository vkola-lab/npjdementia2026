"""
Code executor module for the Discovery Agent.

This module provides a robust Python code executor for LLM-generated code
with timeout mechanisms and persistent global namespaces.
"""

import sys
import io
import contextlib
import signal
import traceback
from typing import Dict, Any, Optional, Union
from ..ui.interface import ui


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


class CodeExecutor:
    """Executes Python code with timeout and error handling."""
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize the code executor.
        
        Args:
            timeout_seconds (int): Maximum execution time in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.global_namespace = {
            '__builtins__': __builtins__,
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'dir': dir,
            'vars': vars,
            'locals': locals,
            'globals': globals,
        }
    
    def _timeout_handler(self, signum, frame):
        """Handle timeout signal."""
        raise TimeoutException(f"Code execution timed out after {self.timeout_seconds} seconds")
    
    def execute_code(self, code: str, local_namespace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Python code with timeout and error handling.
        
        Args:
            code (str): Python code to execute
            local_namespace (Optional[Dict[str, Any]]): Local namespace for execution
        
        Returns:
            Dict[str, Any]: Execution result with success status, output, and error info
        """
        if local_namespace is None:
            local_namespace = {}
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        captured_error = io.StringIO()
        
        try:
            # Redirect stdout and stderr
            sys.stdout = captured_output
            sys.stderr = captured_error
            
            # Set up timeout
            if hasattr(signal, 'SIGALRM'):  # Unix-like systems
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(self.timeout_seconds)
            
            # Execute the code - use global namespace for variable persistence
            exec(code, self.global_namespace)
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Get output
            output = captured_output.getvalue()
            error_output = captured_error.getvalue()
            
            return {
                "success": True,
                "output": output,
                "error": error_output if error_output else None,
                "code": code
            }
            
        except TimeoutException as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "code": code
            }
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {
                "success": False,
                "output": "",
                "error": f"{str(e)}\n\nTraceback:\n{error_traceback}",
                "code": code
            }
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Cancel timeout if still active
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def execute_code_block(self, code_block: str) -> Dict[str, Any]:
        """
        Execute a code block (extracted from LLM response).
        
        Args:
            code_block (str): Code block to execute
        
        Returns:
            Dict[str, Any]: Execution result
        """
        return self.execute_code(code_block)
    
    def get_global_namespace(self) -> Dict[str, Any]:
        """
        Get the current global namespace.
        
        Returns:
            Dict[str, Any]: Global namespace
        """
        return self.global_namespace.copy()
    
    def set_global_variable(self, name: str, value: Any) -> None:
        """
        Set a variable in the global namespace.
        
        Args:
            name (str): Variable name
            value (Any): Variable value
        """
        self.global_namespace[name] = value
    
    def get_global_variable(self, name: str) -> Any:
        """
        Get a variable from the global namespace.
        
        Args:
            name (str): Variable name
        
        Returns:
            Any: Variable value or None if not found
        """
        return self.global_namespace.get(name)
    
    def clear_global_namespace(self) -> None:
        """Clear the global namespace (except built-ins)."""
        # Preserve built-ins and essential functions
        preserved_items = {
            '__builtins__': self.global_namespace.get('__builtins__', {}),
            'print': print,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'type': type,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'dir': dir,
            'vars': vars,
            'locals': locals,
            'globals': globals,
        }
        self.global_namespace.clear()
        self.global_namespace.update(preserved_items)
