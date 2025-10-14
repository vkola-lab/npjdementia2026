"""
Execution utilities for the Discovery Agent.

This module provides utility functions for code execution and related operations.
"""

import re
from typing import Dict, Any, Optional, Union
from .executor import CodeExecutor
from ..ui.interface import ui


def extract_code_blocks(text: str) -> list:
    """
    Extract Python code blocks from text (between ```python and ```).
    
    Args:
        text (str): Text containing code blocks
    
    Returns:
        list: List of extracted code blocks
    """
    # Pattern to match code blocks between ```python and ```
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def execute_code_from_response(code_executor: CodeExecutor, response: str) -> Union[Dict[str, Any], str, None]:
    """
    Execute code blocks from an LLM response.
    
    Args:
        code_executor (CodeExecutor): Code executor instance
        response (str): LLM response containing code blocks
    
    Returns:
        Union[Dict[str, Any], str, None]: Execution result or error message
    """
    try:
        # Extract code blocks from response
        code_blocks = extract_code_blocks(response)
        
        if not code_blocks:
            ui.print_warning("⚠️ No Python code blocks found in response")
            return None
        
        # Execute each code block
        results = []
        for i, code_block in enumerate(code_blocks):
            ui.print_info(f"💻 Executing code block {i + 1}/{len(code_blocks)}...")
            
            # Clean up the code block
            code_block = code_block.strip()
            
            # Execute the code with robust exception handling
            try:
                result = code_executor.execute_code_block(code_block)
                results.append(result)
                
                # Print execution result
                if result["success"]:
                    if result["output"]:
                        ui.print_info(f"✅ Code block {i + 1} executed successfully")
                        ui.print_info(f"📊 Output: {result['output']}")
                    else:
                        ui.print_info(f"✅ Code block {i + 1} executed successfully (no output)")
                    
                    if result["error"]:
                        ui.print_warning(f"⚠️ Code block {i + 1} had warnings: {result['error']}")
                else:
                    ui.print_error(f"❌ Code block {i + 1} failed: {result['error']}")
                    return result["error"]  # Return error string for failed execution
                    
            except Exception as e:
                # Handle unhandled exceptions that could terminate the session
                error_msg = f"Code block {i + 1} execution failed with exception: {str(e)}"
                ui.print_error(f"❌ {error_msg}")
                return error_msg
        
        # Return the last successful result
        if results:
            return results[-1]
        else:
            return None
            
    except Exception as e:
        error_msg = f"Error executing code from response: {str(e)}"
        ui.print_error(error_msg, "code_execution")
        return error_msg


def safe_execute_code(code_executor: CodeExecutor, code: str, 
                     local_namespace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Safely execute code with comprehensive error handling.
    
    Args:
        code_executor (CodeExecutor): Code executor instance
        code (str): Code to execute
        local_namespace (Optional[Dict[str, Any]]): Local namespace
    
    Returns:
        Dict[str, Any]: Execution result
    """
    try:
        return code_executor.execute_code(code, local_namespace)
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Execution error: {str(e)}",
            "code": code
        }


def validate_code_syntax(code: str) -> tuple:
    """
    Validate Python code syntax without executing it.
    
    Args:
        code (str): Code to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def clean_code_output(output: str) -> str:
    """
    Clean and format code output for display.
    
    Args:
        output (str): Raw code output
    
    Returns:
        str: Cleaned output
    """
    if not output:
        return ""
    
    # Remove excessive whitespace
    lines = output.strip().split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    
    # Remove empty lines at the end
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)


def format_execution_result(result: Dict[str, Any]) -> str:
    """
    Format execution result for display.
    
    Args:
        result (Dict[str, Any]): Execution result
    
    Returns:
        str: Formatted result string
    """
    if not result:
        return "No execution result"
    
    status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
    output = result.get("output", "")
    error = result.get("error", "")
    
    formatted = f"Status: {status}\n"
    
    if output:
        formatted += f"Output:\n{clean_code_output(output)}\n"
    
    if error:
        formatted += f"Error:\n{error}\n"
    
    return formatted.strip()
