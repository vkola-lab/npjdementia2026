import io
import time
import traceback
import threading
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr
from .utils import extract_python_code_block
from .ui import ui

class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

class CodeExecutor:
    """Robust code executor for LLM-generated Python code with persistent globals."""

    def __init__(self, timeout_seconds=30):
        self.execution_count = 0
        self.timeout_seconds = timeout_seconds
        # Shared global namespace across all executions
        self.exec_globals = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
            '__doc__': None
        }
        ui.print_info(f"🔧 CodeExecutor initialized with {timeout_seconds}s timeout")

    def extract_code_block(self, text: str) -> str:
        """Extract a single Python code block from text (handles various fence styles)."""
        return extract_python_code_block(text)


    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in an isolated namespace."""
        self.execution_count += 1
        ui.print_info(f"💻 Executing code block")

        # Use a single namespace for both globals and locals
        exec_namespace = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
            '__doc__': None
        }

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        start_time = time.time()
        
        # Thread-based timeout mechanism
        result_container = {'result': None, 'exception': None}
        
        def execute_code_thread():
            """Execute code in a separate thread."""
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, exec_namespace, exec_namespace)
                result_container['result'] = True
            except Exception as e:
                result_container['exception'] = e

        try:
            ui.print_info("🔄 Executing code...")
            ui.print_info(f"⏰ Setting timeout to {self.timeout_seconds}s")
            
            # Start execution in a separate thread
            exec_thread = threading.Thread(target=execute_code_thread)
            exec_thread.daemon = True  # Dies when main thread dies
            exec_thread.start()
            
            # Wait for completion or timeout
            exec_thread.join(timeout=self.timeout_seconds)
            
            if exec_thread.is_alive():
                # Thread is still running - timeout occurred
                ui.print_warning(f"⚠️ Code execution timed out after {self.timeout_seconds}s")
                raise TimeoutError(f"Code execution timed out after {self.timeout_seconds} seconds")
            
            # Check if an exception occurred in the thread
            if result_container['exception'] is not None:
                raise result_container['exception']

            execution_time = time.time() - start_time
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()
            
            ui.print_info(f"✅ Code execution completed in {execution_time:.2f}s")
            if output:
                ui.print_info(f"📊 Output: {output[:100]}...")
            if error:
                ui.print_warning(f"⚠️ Warnings: {error[:100]}...")
            
            return {
                "success": True,
                "output": output,
                "error": error,
                "execution_time": execution_time,
                "code": ""
            }

        except TimeoutError as e:
            execution_time = time.time() - start_time
            ui.print_error(f"Code execution timed out after {self.timeout_seconds}s: {str(e)}", "code_execution")
            return {
                "success": False,
                "output": f"Code execution timed out after {self.timeout_seconds} seconds",
                "error": f"TimeoutError: {str(e)}",
                "execution_time": execution_time,
                "code": ""
            }
        except Exception as e:
            execution_time = time.time() - start_time
            executed_output = str(traceback.format_exc())
            ui.print_error(f"Code execution failed: {str(e)}", "code_execution")
            return {
                "success": False,
                "output": executed_output + stdout_capture.getvalue(),
                "error": executed_output,
                "execution_time": execution_time,
                "code": ""
            }


    def execute_from_text(self, text: str) -> Dict[str, Any]:
        """Extract and execute a single code block from text."""
        ui.print_info("🔍 Extracting code block from text...")
        code_block = self.extract_code_block(text)
        if not code_block:
            ui.print_warning("⚠️ No code block found in text")
            return {
                "success": False,
                "output": "",
                "error": "No code block found.",
                "execution_time": 0,
                "code": ""
            }
        
        ui.print_info(f"📝 Found code block: {len(code_block)} characters")
        result = self.execute_code(code_block)
        result["code"] = code_block
        return result
