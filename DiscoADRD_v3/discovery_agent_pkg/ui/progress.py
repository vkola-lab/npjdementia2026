"""
Progress tracking and display components for the Discovery Agent UI.

This module provides production-level progress indicators, status displays,
and real-time monitoring capabilities.
"""

import time
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .colors import Colors, Styling, default_palette

class ProgressTracker:
    """Advanced progress tracking with multiple metrics."""
    
    def __init__(self, total_items: int, title: str = "Progress"):
        """Initialize progress tracker."""
        self.total_items = total_items
        self.current_item = 0
        self.title = title
        self.start_time = time.time()
        self.items_completed = 0
        self.items_failed = 0
        self.current_phase = "Initializing"
        self.phases = []
        self.phase_start_time = None
        self.metrics = {}
        
    def start_phase(self, phase_name: str):
        """Start a new phase."""
        if self.phase_start_time:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        self.phases.append({
            "name": phase_name,
            "start_time": self.phase_start_time,
            "end_time": None,
            "duration": None
        })
    
    def end_phase(self):
        """End current phase."""
        if self.phase_start_time:
            end_time = time.time()
            duration = end_time - self.phase_start_time
            
            # Update the last phase
            if self.phases:
                self.phases[-1]["end_time"] = end_time
                self.phases[-1]["duration"] = duration
            
            self.phase_start_time = None
    
    def update(self, item_name: str = "", success: bool = True):
        """Update progress."""
        self.current_item += 1
        if success:
            self.items_completed += 1
        else:
            self.items_failed += 1
    
    def set_metric(self, key: str, value: Any):
        """Set a custom metric."""
        self.metrics[key] = value
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion."""
        if self.current_item == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        rate = self.current_item / elapsed
        remaining = self.total_items - self.current_item
        
        return remaining / rate if rate > 0 else None
    
    def get_completion_percentage(self) -> float:
        """Get completion percentage."""
        return (self.current_item / self.total_items) * 100 if self.total_items > 0 else 0

class ProgressDisplay:
    """Production-level progress display with multiple formats."""
    
    def __init__(self, palette=None):
        """Initialize progress display."""
        self.palette = palette or default_palette
        self.last_update_time = 0
        self.update_interval = 0.1  # Update every 100ms
    
    def should_update(self) -> bool:
        """Check if display should update."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    def display_progress(self, tracker: ProgressTracker, verbose: bool = True) -> None:
        """Display progress with multiple formats."""
        if not verbose or not self.should_update():
            return
        
        # Clear line and move cursor to beginning
        sys.stdout.write('\r\033[K')
        
        # Create progress display
        display = self._create_progress_display(tracker)
        sys.stdout.write(display)
        sys.stdout.flush()
    
    def _create_progress_display(self, tracker: ProgressTracker) -> str:
        """Create comprehensive progress display."""
        percentage = tracker.get_completion_percentage()
        elapsed = tracker.get_elapsed_time()
        eta = tracker.get_eta()
        
        # Progress bar
        progress_bar = Styling.progress_bar(
            tracker.current_item, 
            tracker.total_items, 
            width=30,
            fill_char="█",
            empty_char="░"
        )
        
        # Time information
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta) if eta else "Unknown"
        
        # Status indicators
        success_indicator = f"{self.palette.success}✓{tracker.items_completed}{Colors.RESET}"
        failed_indicator = f"{self.palette.error}✗{tracker.items_failed}{Colors.RESET}" if tracker.items_failed > 0 else ""
        
        # Current phase
        phase_indicator = f"{self.palette.info}Phase: {tracker.current_phase}{Colors.RESET}"
        
        # Build display
        display_parts = [
            f"{self.palette.progress}{progress_bar}{Colors.RESET}",
            f"{self.palette.muted}({percentage:.1f}%){Colors.RESET}",
            f"{self.palette.muted}Elapsed: {elapsed_str}{Colors.RESET}",
            f"{self.palette.muted}ETA: {eta_str}{Colors.RESET}",
            success_indicator,
            failed_indicator,
            phase_indicator
        ]
        
        return " ".join(filter(None, display_parts))
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds is None:
            return "Unknown"
        
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def display_summary(self, tracker: ProgressTracker, verbose: bool = True) -> None:
        """Display final progress summary."""
        if not verbose:
            return
        
        print()  # New line after progress bar
        
        # Summary box
        summary_lines = [
            f"Completed: {tracker.items_completed}/{tracker.total_items}",
            f"Failed: {tracker.items_failed}",
            f"Success Rate: {(tracker.items_completed / tracker.total_items * 100):.1f}%" if tracker.total_items > 0 else "Success Rate: 0%",
            f"Total Time: {self._format_time(tracker.get_elapsed_time())}"
        ]
        
        # Add phase breakdown
        if tracker.phases:
            summary_lines.append("")
            summary_lines.append("Phase Breakdown:")
            for phase in tracker.phases:
                if phase["duration"]:
                    duration_str = self._format_time(phase["duration"])
                    summary_lines.append(f"  • {phase['name']}: {duration_str}")
        
        # Add custom metrics
        if tracker.metrics:
            summary_lines.append("")
            summary_lines.append("Metrics:")
            for key, value in tracker.metrics.items():
                summary_lines.append(f"  • {key}: {value}")
        
        summary_text = "\n".join(summary_lines)
        summary_box = Styling.box(summary_text, self.palette.success)
        print(summary_box)

class StatusDisplay:
    """Real-time status display for system components."""
    
    def __init__(self, palette=None):
        """Initialize status display."""
        self.palette = palette or default_palette
        self.statuses = {}
        self.last_update = 0
    
    def set_status(self, component: str, status: str, message: str = ""):
        """Set status for a component."""
        self.statuses[component] = {
            "status": status,
            "message": message,
            "timestamp": time.time()
        }
    
    def display_status(self, verbose: bool = True) -> None:
        """Display current status of all components."""
        if not verbose or not self.statuses:
            return
        
        print(f"\n{self.palette.info}System Status:{Colors.RESET}")
        
        for component, info in self.statuses.items():
            status_color = self._get_status_color(info["status"])
            indicator = Styling.status_indicator(info["status"], status_color)
            
            status_line = f"  {indicator} {component}: {info['message']}"
            print(status_line)
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status."""
        status_colors = {
            "success": self.palette.success,
            "error": self.palette.error,
            "warning": self.palette.warning,
            "info": self.palette.info,
            "processing": self.palette.status_processing,
            "pending": self.palette.muted,
            "approved": self.palette.status_approved,
            "rejected": self.palette.status_rejected
        }
        
        return status_colors.get(status.lower(), self.palette.muted)

class ExperimentProgress:
    """Specialized progress tracking for experiments."""
    
    def __init__(self, total_runs: int, total_hypotheses: int, palette=None):
        """Initialize experiment progress tracker."""
        self.palette = palette or default_palette
        self.total_runs = total_runs
        self.total_hypotheses = total_hypotheses
        self.current_run = 0
        self.current_hypothesis = 0
        self.hypotheses_approved = 0
        self.hypotheses_rejected = 0
        self.start_time = time.time()
        self.run_times = []
        self.hypothesis_times = []
        
    def start_run(self, run_number: int):
        """Start a new run."""
        self.current_run = run_number
        self.run_start_time = time.time()
    
    def end_run(self):
        """End current run."""
        if hasattr(self, 'run_start_time'):
            run_time = time.time() - self.run_start_time
            self.run_times.append(run_time)
    
    def start_hypothesis(self, hypothesis_number: int):
        """Start a new hypothesis."""
        self.current_hypothesis = hypothesis_number
        self.hypothesis_start_time = time.time()
    
    def end_hypothesis(self, approved: bool):
        """End current hypothesis."""
        if hasattr(self, 'hypothesis_start_time'):
            hypothesis_time = time.time() - self.hypothesis_start_time
            self.hypothesis_times.append(hypothesis_time)
            
            if approved:
                self.hypotheses_approved += 1
            else:
                self.hypotheses_rejected += 1
    
    def display_experiment_progress(self, verbose: bool = True) -> None:
        """Display experiment-specific progress."""
        if not verbose:
            return
        
        # Overall progress
        total_progress = (self.current_run - 1) * self.total_hypotheses + self.current_hypothesis
        total_items = self.total_runs * self.total_hypotheses
        
        if total_items > 0:
            percentage = (total_progress / total_items) * 100
        else:
            percentage = 0
        
        # Progress bar
        progress_bar = Styling.progress_bar(total_progress, total_items, width=40)
        
        # Statistics
        approval_rate = (self.hypotheses_approved / (self.hypotheses_approved + self.hypotheses_rejected) * 100) if (self.hypotheses_approved + self.hypotheses_rejected) > 0 else 0
        
        # Display
        display_parts = [
            f"{self.palette.progress}{progress_bar}{Colors.RESET}",
            f"{self.palette.muted}({percentage:.1f}%){Colors.RESET}",
            f"{self.palette.info}Run: {self.current_run}/{self.total_runs}{Colors.RESET}",
            f"{self.palette.info}Hypothesis: {self.current_hypothesis}/{self.total_hypotheses}{Colors.RESET}",
            f"{self.palette.success}Approved: {self.hypotheses_approved}{Colors.RESET}",
            f"{self.palette.error}Rejected: {self.hypotheses_rejected}{Colors.RESET}",
            f"{self.palette.muted}Rate: {approval_rate:.1f}%{Colors.RESET}"
        ]
        
        sys.stdout.write('\r\033[K')
        sys.stdout.write(" ".join(display_parts))
        sys.stdout.flush()
    
    def display_experiment_summary(self, verbose: bool = True) -> None:
        """Display final experiment summary."""
        if not verbose:
            return
        
        print()  # New line after progress bar
        
        # Calculate statistics
        total_hypotheses_processed = self.hypotheses_approved + self.hypotheses_rejected
        approval_rate = (self.hypotheses_approved / total_hypotheses_processed * 100) if total_hypotheses_processed > 0 else 0
        
        avg_run_time = sum(self.run_times) / len(self.run_times) if self.run_times else 0
        avg_hypothesis_time = sum(self.hypothesis_times) / len(self.hypothesis_times) if self.hypothesis_times else 0
        
        total_time = time.time() - self.start_time
        
        # Summary box
        summary_lines = [
            f"Experiment Summary",
            f"",
            f"Runs Completed: {self.current_run}/{self.total_runs}",
            f"Hypotheses Processed: {total_hypotheses_processed}",
            f"Approved: {self.hypotheses_approved}",
            f"Rejected: {self.hypotheses_rejected}",
            f"Approval Rate: {approval_rate:.1f}%",
            f"",
            f"Timing:",
            f"  Total Time: {self._format_time(total_time)}",
            f"  Avg Run Time: {self._format_time(avg_run_time)}",
            f"  Avg Hypothesis Time: {self._format_time(avg_hypothesis_time)}"
        ]
        
        summary_text = "\n".join(summary_lines)
        summary_box = Styling.box(summary_text, self.palette.primary)
        print(summary_box)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes:.0f}m {seconds:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
