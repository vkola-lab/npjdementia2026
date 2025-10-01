"""
Utility functions for the Discovery Agent.

This module contains utility functions for system cleanup,
signal handling, and other helper functions.
"""

import os
import subprocess
import sys
import signal
import logging
import emoji
import re
import time

logger = logging.getLogger(__name__)


def cleanup_gpu_processes():
    """Clean up GPU processes for the current user."""
    print("🧹 Cleaning up GPU python processes for CUDA_VISIBLE_DEVICES...")
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not cuda_devices:
        print("ℹ️ CUDA_VISIBLE_DEVICES not set. No GPU processes to clean.")
        return
    
    device_ids = [d.strip() for d in cuda_devices.split(',') if d.strip().isdigit()]
    try:
        # Get all GPU processes from nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], 
            capture_output=True, text=True
        )
        killed = []
        
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 2:
                continue
            
            pid, gpu_uuid = parts
            
            # Check if this GPU is in CUDA_VISIBLE_DEVICES
            # Map device id to uuid
            uuid_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,gpu_uuid', '--format=csv,noheader'], 
                capture_output=True, text=True
            )
            uuid_map = {
                u.split(',')[0].strip(): u.split(',')[1].strip() 
                for u in uuid_result.stdout.splitlines()
            }
            allowed_uuids = [uuid_map.get(d) for d in device_ids if d in uuid_map]
            
            if gpu_uuid in allowed_uuids:
                # Check if process is python and belongs to current user
                ps_result = subprocess.run(
                    ['ps', '-p', pid, '-o', 'user=', '-o', 'comm='], 
                    capture_output=True, text=True
                )
                for line in ps_result.stdout.splitlines():
                    if os.getlogin() in line and 'python' in line:
                        try:
                            os.kill(int(pid), 9)
                            killed.append(pid)
                        except Exception as e:
                            print(f"❌ Error killing PID {pid}: {e}")
        
        if killed:
            print(f"✅ Killed GPU python processes: {', '.join(killed)}")
        else:
            print(f"ℹ️ No GPU python processes found for user {os.getlogin()} on CUDA_VISIBLE_DEVICES={cuda_devices}.")
            
    except Exception as e:
        print(f"❌ Error cleaning GPU processes: {e}")


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    print(f"🛑 Interrupt received (signal {sig}). Cleaning up...")
    cleanup_gpu_processes()
    sys.exit(0)


def sigtstp_handler(sig, frame):
    """Handle Ctrl+Z (SIGTSTP) signal."""
    print(f"⏸️ Terminal stop signal received (signal {sig}). Cleaning up...")
    cleanup_gpu_processes()
    sys.exit(0)


def execute_code_from_response(code_executor, response):
    """Execute code from agent response - returns raw result."""
    try:
        print("🚀 Starting code execution from response...")
        result = code_executor.execute_from_text(response)
        return result
        
    except Exception as e:
        return f"Code execution error: {str(e)}"


def cleanup_resources():
    """Cleanup resources and processes."""
    try:
        cleanup_gpu_processes()
    except Exception as e:
        print(f"Error during cleanup: {e}")


def create_log_file(prefix="session_log", extension="txt"):
    """Create responses directory and return a timestamped log file path."""
    responses_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "responses")
    os.makedirs(responses_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(responses_dir, f"{prefix}_{timestamp}.{extension}")


def remove_ansi_color_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def extract_python_code_block(text: str) -> str:
    """Extract a single Python code block from text (handles various fence styles)."""
    pattern = re.compile(r"```(?:python|Python)?\s*(.*?)```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""




# Additional utility functions can be added here as needed