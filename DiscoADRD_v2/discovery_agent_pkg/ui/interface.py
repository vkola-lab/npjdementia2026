"""
Enhanced user interface module for the Discovery Agent.

This module provides a production-level user interface with advanced formatting,
progress tracking, and beautiful output display.
"""

import os
import sys
import time
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
# Try to import enhanced UI components, fall back to basic functionality if not available
try:
    from .colors import Colors, Styling, default_palette, ColorTheme
    from .progress import ProgressTracker, ProgressDisplay, StatusDisplay, ExperimentProgress
    from .formatting import TableFormatter, ReportFormatter, BannerFormatter, CodeFormatter, StatusFormatter
    ENHANCED_UI_AVAILABLE = True
except ImportError:
    # Fallback to basic functionality
    ENHANCED_UI_AVAILABLE = False
    
    # Create minimal fallback classes
    class Colors:
        RESET = "\033[0m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_RED = "\033[91m"
        BRIGHT_WHITE = "\033[97m"
        BRIGHT_BLACK = "\033[90m"
        BRIGHT_MAGENTA = "\033[95m"
        WHITE = "\033[37m"
    
    class Styling:
        @staticmethod
        def box(text, color):
            return f"{color}{text}{Colors.RESET}"
    
    class ColorTheme:
        DEFAULT = "default"
    
    class default_palette:
        def __init__(self):
            self.primary = Colors.BRIGHT_CYAN
            self.secondary = Colors.BRIGHT_BLUE
            self.success = Colors.BRIGHT_GREEN
            self.warning = Colors.BRIGHT_YELLOW
            self.error = Colors.BRIGHT_RED
            self.info = Colors.BRIGHT_BLUE
            self.muted = Colors.BRIGHT_BLACK
            self.highlight = Colors.BRIGHT_WHITE
            self.accent = Colors.BRIGHT_MAGENTA
            self.banner = Colors.BRIGHT_CYAN
            self.separator = Colors.BRIGHT_BLACK
            self.progress = Colors.BRIGHT_MAGENTA
            self.agent_scientist = Colors.BRIGHT_GREEN
            self.agent_critic = Colors.BRIGHT_RED
            self.agent_planner = Colors.BRIGHT_BLUE
            self.hypothesis = Colors.BRIGHT_WHITE
            self.code = Colors.BRIGHT_BLACK
            self.table_header = Colors.BRIGHT_BLUE
            self.table_row = Colors.WHITE
            self.status_approved = Colors.BRIGHT_GREEN
            self.status_rejected = Colors.BRIGHT_RED
            self.status_processing = Colors.BRIGHT_YELLOW
    
    # Create minimal fallback classes for other components
    class ProgressTracker:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, *args, **kwargs):
            pass
    
    class ProgressDisplay:
        def __init__(self, *args, **kwargs):
            pass
        def display_progress(self, *args, **kwargs):
            pass
        def display_summary(self, *args, **kwargs):
            pass
    
    class StatusDisplay:
        def __init__(self, *args, **kwargs):
            pass
        def set_status(self, *args, **kwargs):
            pass
        def display_status(self, *args, **kwargs):
            pass
    
    class ExperimentProgress:
        def __init__(self, *args, **kwargs):
            pass
        def start_run(self, *args, **kwargs):
            pass
        def end_run(self, *args, **kwargs):
            pass
        def start_hypothesis(self, *args, **kwargs):
            pass
        def end_hypothesis(self, *args, **kwargs):
            pass
        def display_experiment_progress(self, *args, **kwargs):
            pass
        def display_experiment_summary(self, *args, **kwargs):
            pass
    
    class TableFormatter:
        def __init__(self, *args, **kwargs):
            pass
        def format_table(self, *args, **kwargs):
            return ""
    
    class ReportFormatter:
        def __init__(self, *args, **kwargs):
            pass
        def format_experiment_report(self, *args, **kwargs):
            return ""
    
    class BannerFormatter:
        def __init__(self, *args, **kwargs):
            pass
        def format_main_banner(self, *args, **kwargs):
            return "🔬 Discovery Agent - AI-Powered Scientific Discovery System"
        def format_section_banner(self, *args, **kwargs):
            return ""
        def format_subsection_banner(self, *args, **kwargs):
            return ""
    
    class CodeFormatter:
        def __init__(self, *args, **kwargs):
            pass
        def format_code_block(self, *args, **kwargs):
            return ""
        def format_output_block(self, *args, **kwargs):
            return ""
    
    class StatusFormatter:
        def __init__(self, *args, **kwargs):
            pass
        def format_status_badge(self, *args, **kwargs):
            return ""
        def format_progress_indicator(self, *args, **kwargs):
            return ""

def remove_ansi_color_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class UI:
    """Enhanced user interface for the Discovery Agent."""
    
    def __init__(self, theme: ColorTheme = ColorTheme.DEFAULT):
        """Initialize the enhanced UI."""
        self.session_log = []
        self.colors_enabled = sys.stdout.isatty()  # Enable colors if terminal supports it
        self.enhanced_ui_available = ENHANCED_UI_AVAILABLE
        
        # Initialize palette correctly based on availability
        if self.enhanced_ui_available:
            self.palette = default_palette
        else:
            self.palette = default_palette()
        
        self.start_time = time.time()
        
        # Initialize formatters (with fallback support)
        self.table_formatter = TableFormatter(self.palette)
        self.report_formatter = ReportFormatter(self.palette)
        self.banner_formatter = BannerFormatter(self.palette)
        self.code_formatter = CodeFormatter(self.palette)
        self.status_formatter = StatusFormatter(self.palette)
        
        # Initialize progress components (with fallback support)
        self.progress_display = ProgressDisplay(self.palette)
        self.status_display = StatusDisplay(self.palette)
        
        # Current progress trackers
        self.current_progress_tracker = None
        self.current_experiment_progress = None
        
        # Log saving state
        self.log_saved = False
        self.log_file = None
        self.log_file_path = None
        
        # Initialize real-time logging
        self._init_realtime_logging()
        
        # Register cleanup function to save log on exit
        import atexit
        atexit.register(self.close_log)
    
    def _init_realtime_logging(self):
        """Initialize real-time logging to a file."""
        try:
            import os
            import time
            
            # Create log file path
            responses_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "analysis", "responses")
            os.makedirs(responses_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_file_path = os.path.join(responses_dir, f"session_log_{timestamp}.txt")
            
            # Open log file for real-time writing
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            
        except Exception as e:
            print(f"❌ Failed to initialize real-time logging: {str(e)}")
            self.log_file = None
            self.log_file_path = None
    
    def _print_colored(self, text: str, color_code: str, verbose: bool = True) -> None:
        """
        Print colored text if colors are enabled and verbose is True.
        
        Args:
            text (str): Text to print
            color_code (str): ANSI color code
            verbose (bool): Whether to print (verbose mode)
        """
        if not verbose:
            return
        
        if self.colors_enabled:
            print(f"{color_code}{text}\033[0m")
        else:
            print(text)
        
        # Log to session
        clean_text = remove_ansi_color_codes(text)
        self.session_log.append(clean_text)
        
        # Write to real-time log file
        if self.log_file:
            try:
                self.log_file.write(clean_text + '\n')
                self.log_file.flush()  # Ensure immediate write
            except Exception as e:
                # If real-time logging fails, continue without it
                pass
    
    def print_info(self, message: str, verbose: bool = True) -> None:
        """
        Print an info message with enhanced formatting.
        
        Args:
            message (str): Message to print
            verbose (bool): Whether to print (verbose mode)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"{self.palette.muted}[{timestamp}]{Colors.RESET} {self.palette.info}ℹ️  {message}{Colors.RESET}"
        self._print_colored(formatted_message, "", verbose)
    
    def print_success(self, message: str, verbose: bool = True) -> None:
        """
        Print a success message with enhanced formatting.
        
        Args:
            message (str): Message to print
            verbose (bool): Whether to print (verbose mode)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"{self.palette.muted}[{timestamp}]{Colors.RESET} {self.palette.success}✅ {message}{Colors.RESET}"
        self._print_colored(formatted_message, "", verbose)
    
    def print_warning(self, message: str, verbose: bool = True) -> None:
        """
        Print a warning message with enhanced formatting.
        
        Args:
            message (str): Message to print
            verbose (bool): Whether to print (verbose mode)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"{self.palette.muted}[{timestamp}]{Colors.RESET} {self.palette.warning}⚠️  {message}{Colors.RESET}"
        self._print_colored(formatted_message, "", verbose)
    
    def print_error(self, message: str, context: str = "", verbose: bool = True) -> None:
        """
        Print an error message with enhanced formatting.
        
        Args:
            message (str): Error message to print
            context (str): Error context
            verbose (bool): Whether to print (verbose mode)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_msg = f"{self.palette.muted}[{timestamp}]{Colors.RESET} {self.palette.error}❌ {message}{Colors.RESET}"
        if context:
            error_msg += f" {self.palette.muted}(Context: {context}){Colors.RESET}"
        self._print_colored(error_msg, "", verbose)
    
    def print_loading(self, message: str, verbose: bool = True) -> None:
        """
        Print a loading message.
        
        Args:
            message (str): Loading message to print
            verbose (bool): Whether to print (verbose mode)
        """
        self._print_colored(f"⏳ {message}", "\033[96m", verbose)
    
    def print_separator(self, char: str = "=", length: int = 50, verbose: bool = True) -> None:
        """
        Print a separator line.
        
        Args:
            char (str): Character to use for separator
            length (int): Length of separator
            verbose (bool): Whether to print (verbose mode)
        """
        separator = char * length
        self._print_colored(separator, "\033[90m", verbose)
    
    def print_progress(self, current: int, total: int, verbose: bool = True) -> None:
        """
        Print progress information.
        
        Args:
            current (int): Current progress
            total (int): Total items
            verbose (bool): Whether to print (verbose mode)
        """
        progress_msg = f"📊 Progress: {current}/{total} ({current/total*100:.1f}%)"
        self._print_colored(progress_msg, "\033[95m", verbose)
    
    def print_hypothesis_list(self, hypotheses: List[str], generated: bool = True, verbose: bool = True) -> None:
        """
        Print a list of hypotheses.
        
        Args:
            hypotheses (List[str]): List of hypotheses to print
            generated (bool): Whether hypotheses were generated or direct
            verbose (bool): Whether to print (verbose mode)
        """
        if not verbose:
            return
        
        prefix = "🔬 Generated hypotheses:" if generated else "📝 Direct hypotheses:"
        self._print_colored(prefix, "\033[94m", verbose)
        
        for i, hypothesis in enumerate(hypotheses, 1):
            self._print_colored(f"   {i}. {hypothesis}", "\033[97m", verbose)
    
    def print_approval_status(self, status: str, verbose: bool = True) -> None:
        """
        Print hypothesis approval status.
        
        Args:
            status (str): Status of hypothesis ("HYPOTHESIS VERIFIED", "HYPOTHESIS NOT VERIFIED", "NOT APPLICABLE")
            verbose (bool): Whether to print (verbose mode)
        """
        if status == "HYPOTHESIS VERIFIED":
            self.print_success("🎉 Hypothesis VERIFIED!", verbose)
        elif status == "NOT APPLICABLE":
            self.print_warning("⚠️ Hypothesis NOT APPLICABLE", verbose)
        else:  # HYPOTHESIS NOT VERIFIED
            self.print_warning("⚠️ Hypothesis NOT VERIFIED", verbose)
    
    def print_agent_response(self, agent_name: str, response: str, code_output: str = "", verbose: bool = True) -> None:
        """
        Print an agent response with formatting.
        
        Args:
            agent_name (str): Name of the agent
            response (str): Agent's response
            code_output (str): Code execution output
            verbose (bool): Whether to print (verbose mode)
        """
        if not verbose:
            return
        
        # Print agent header
        self._print_colored(f"\n🤖 {agent_name} Response:", "\033[95m", verbose)
        
        # Print response
        self._print_colored(response, "\033[97m", verbose)
        
        # Print code output if present
        if code_output:
            self._print_colored(f"\n💻 Code Output:", "\033[96m", verbose)
            self._print_colored(code_output, "\033[90m", verbose)
        
        print()  # Add spacing
    
    def get_user_input(self, prompt: str = "Enter your input: ") -> str:
        """
        Get user input from the command line.
        
        Args:
            prompt (str): Prompt to display
        
        Returns:
            str: User input
        """
        try:
            return input(f"🔍 {prompt}")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    def get_user_choice(self, prompt: str, choices: List[str]) -> str:
        """
        Get user choice from a list of options.
        
        Args:
            prompt (str): Prompt to display
            choices (List[str]): List of available choices
        
        Returns:
            str: User's choice
        """
        print(f"🔍 {prompt}")
        for i, choice in enumerate(choices, 1):
            print(f"   {i}. {choice}")
        
        while True:
            try:
                choice_num = int(input("Enter your choice (number): "))
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def confirm_action(self, message: str) -> bool:
        """
        Ask user to confirm an action.
        
        Args:
            message (str): Confirmation message
        
        Returns:
            bool: True if user confirms, False otherwise
        """
        while True:
            response = input(f"❓ {message} (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def save_session_log(self, filepath: str) -> None:
        """
        Save the session log to a file.
        
        Args:
            filepath (str): Path to save the log file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.session_log))
            self.print_success(f"Session log saved to {filepath}")
        except Exception as e:
            self.print_error(f"Failed to save session log: {str(e)}")
    
    def clear_session_log(self) -> None:
        """Clear the current session log."""
        self.session_log = []
    
    def get_session_log(self) -> List[str]:
        """
        Get the current session log.
        
        Returns:
            List[str]: List of logged messages
        """
        return self.session_log.copy()
    
    def print_table(self, data: List[Dict[str, Any]], headers: Optional[List[str]] = None, verbose: bool = True) -> None:
        """
        Print data in a table format.
        
        Args:
            data (List[Dict[str, Any]]): Data to display
            headers (Optional[List[str]]): Column headers
            verbose (bool): Whether to print (verbose mode)
        """
        if not verbose or not data:
            return
        
        if headers is None:
            headers = list(data[0].keys())
        
        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)
            for row in data:
                value = str(row.get(header, ""))
                col_widths[header] = max(col_widths[header], len(value))
        
        # Print header
        header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
        self._print_colored(header_row, "\033[94m", verbose)
        
        # Print separator
        separator = "-+-".join("-" * col_widths[header] for header in headers)
        self._print_colored(separator, "\033[90m", verbose)
        
        # Print data rows
        for row in data:
            data_row = " | ".join(str(row.get(header, "")).ljust(col_widths[header]) for header in headers)
            self._print_colored(data_row, "\033[97m", verbose)
    
    def print_banner(self, verbose: bool = True) -> None:
        """
        Print the enhanced Discovery Agent banner.
        
        Args:
            verbose (bool): Whether to print (verbose mode)
        """
        if not verbose:
            return
        
        if self.enhanced_ui_available:
            banner = self.banner_formatter.format_main_banner(
                title="🔬 Discovery Agent",
                subtitle="AI-Powered Scientific Discovery System",
                version="1.0.0"
            )
            print(banner)
        else:
            # Fallback to simple banner
            banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🔬 Discovery Agent 🔬                              ║
║                    AI-Powered Scientific Discovery System                    ║
║                                                                              ║
║  Multi-Agent System: Scientist • Critic • Planning Agent                    ║
║  Hypothesis-Driven Research • Data Analysis • Statistical Testing           ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """
            self._print_colored(banner.strip(), self.palette.banner, verbose)
        
        print()  # Add spacing
    
    def close_log(self) -> None:
        """Close the session log and real-time log file."""
        if self.log_saved:
            return
        
        try:
            # Close real-time log file if it exists
            if self.log_file:
                self.log_file.close()
                self.log_file = None
                if self.log_file_path:
                    print(f"📝 Real-time session log saved to: {self.log_file_path}")
            
            # If no real-time log file was created, create one from session log
            elif self.session_log:
                import os
                import time
                
                # Create log file path directly
                responses_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "analysis", "responses")
                os.makedirs(responses_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file_path = os.path.join(responses_dir, f"session_log_{timestamp}.txt")
                
                # Save the session log
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(self.session_log))
                
                print(f"📝 Session log saved to: {log_file_path}")
            
            self.log_saved = True
            
        except Exception as e:
            print(f"❌ Failed to save session log: {str(e)}")
    
    # ==================== NEW PRODUCTION-LEVEL METHODS ====================
    
    def print_section_banner(self, title: str, verbose: bool = True) -> None:
        """Print a section banner."""
        if not verbose:
            return
        if self.enhanced_ui_available:
            banner = self.banner_formatter.format_section_banner(title)
            print(banner)
        else:
            # Fallback
            print(f"\n{self.palette.primary}{'=' * 60}{Colors.RESET}")
            print(f"{self.palette.primary}{title.center(60)}{Colors.RESET}")
            print(f"{self.palette.primary}{'=' * 60}{Colors.RESET}\n")
    
    def print_subsection_banner(self, title: str, verbose: bool = True) -> None:
        """Print a subsection banner."""
        if not verbose:
            return
        if self.enhanced_ui_available:
            banner = self.banner_formatter.format_subsection_banner(title)
            print(banner)
        else:
            # Fallback
            print(f"\n{self.palette.secondary}{'-' * 40}{Colors.RESET}")
            print(f"{self.palette.secondary}{title.center(40)}{Colors.RESET}")
            print(f"{self.palette.secondary}{'-' * 40}{Colors.RESET}\n")
    
    def print_enhanced_table(self, data: List[Dict[str, Any]], 
                           headers: Optional[List[str]] = None,
                           style: str = "default",
                           title: str = "",
                           verbose: bool = True) -> None:
        """Print an enhanced table with various styles."""
        if not verbose or not data:
            return
        
        if title:
            self.print_subsection_banner(title, verbose)
        
        table = self.table_formatter.format_table(data, headers, style)
        print(table)
        print()
    
    def print_status_badge(self, status: str, text: str = "", verbose: bool = True) -> None:
        """Print a status badge."""
        if not verbose:
            return
        badge = self.status_formatter.format_status_badge(status, text)
        print(badge)
    
    def print_progress_indicator(self, current: int, total: int, 
                               label: str = "", verbose: bool = True) -> None:
        """Print a progress indicator."""
        if not verbose:
            return
        indicator = self.status_formatter.format_progress_indicator(current, total, label)
        print(f"\r{indicator}", end="", flush=True)
    
    def print_code_block(self, code: str, language: str = "python", 
                        title: str = "Code", verbose: bool = True) -> None:
        """Print a formatted code block."""
        if not verbose:
            return
        
        if title:
            self.print_subsection_banner(title, verbose)
        
        code_block = self.code_formatter.format_code_block(code, language)
        print(code_block)
        print()
    
    def print_output_block(self, output: str, title: str = "Output", verbose: bool = True) -> None:
        """Print a formatted output block."""
        if not verbose:
            return
        
        output_block = self.code_formatter.format_output_block(output, title)
        print(output_block)
        print()
    
    def print_experiment_report(self, experiment_data: Dict[str, Any], verbose: bool = True) -> None:
        """Print a comprehensive experiment report."""
        if not verbose:
            return
        
        report = self.report_formatter.format_experiment_report(experiment_data)
        print(report)
        print()
    
    def start_progress_tracking(self, total_items: int, title: str = "Progress") -> ProgressTracker:
        """Start progress tracking."""
        self.current_progress_tracker = ProgressTracker(total_items, title)
        return self.current_progress_tracker
    
    def update_progress(self, item_name: str = "", success: bool = True, verbose: bool = True) -> None:
        """Update current progress."""
        if self.current_progress_tracker:
            self.current_progress_tracker.update(item_name, success)
            self.progress_display.display_progress(self.current_progress_tracker, verbose)
    
    def end_progress_tracking(self, verbose: bool = True) -> None:
        """End progress tracking and show summary."""
        if self.current_progress_tracker:
            self.progress_display.display_summary(self.current_progress_tracker, verbose)
            self.current_progress_tracker = None
    
    def start_experiment_progress(self, total_runs: int, total_hypotheses: int) -> ExperimentProgress:
        """Start experiment progress tracking."""
        self.current_experiment_progress = ExperimentProgress(total_runs, total_hypotheses, self.palette)
        return self.current_experiment_progress
    
    def update_experiment_progress(self, run_number: int = None, hypothesis_number: int = None, 
                                 approved: bool = None, verbose: bool = True) -> None:
        """Update experiment progress."""
        if self.current_experiment_progress:
            if run_number is not None:
                self.current_experiment_progress.start_run(run_number)
            if hypothesis_number is not None:
                self.current_experiment_progress.start_hypothesis(hypothesis_number)
            if approved is not None:
                self.current_experiment_progress.end_hypothesis(approved)
            
            self.current_experiment_progress.display_experiment_progress(verbose)
    
    def end_experiment_progress(self, verbose: bool = True) -> None:
        """End experiment progress and show summary."""
        if self.current_experiment_progress:
            self.current_experiment_progress.display_experiment_summary(verbose)
            self.current_experiment_progress = None
    
    def set_system_status(self, component: str, status: str, message: str = ""):
        """Set system component status."""
        self.status_display.set_status(component, status, message)
    
    def display_system_status(self, verbose: bool = True) -> None:
        """Display current system status."""
        self.status_display.display_status(verbose)
    
    def print_agent_response_enhanced(self, agent_name: str, response: str, 
                                    code_output: str = "", verbose: bool = True) -> None:
        """Print enhanced agent response with better formatting."""
        if not verbose:
            return
        
        # Agent header with styling
        agent_colors = {
            "Scientist": self.palette.agent_scientist,
            "Critic": self.palette.agent_critic,
            "Planning Agent": self.palette.agent_planner
        }
        
        agent_color = agent_colors.get(agent_name, self.palette.primary)
        header = f"\n{agent_color}🤖 {agent_name} Response{Colors.RESET}"
        print(header)
        print(f"{agent_color}{'─' * (len(agent_name) + 12)}{Colors.RESET}")
        
        # Response content
        print(f"{self.palette.highlight}{response}{Colors.RESET}")
        
        # Code output if present
        if code_output:
            print()
            self.print_output_block(code_output, "Code Execution Output", verbose)
        
        print()  # Add spacing
    
    def print_hypothesis_enhanced(self, hypotheses: List[str], generated: bool = True, 
                                verbose: bool = True) -> None:
        """Print enhanced hypothesis list."""
        if not verbose or not hypotheses:
            return
        
        prefix = "🔬 Generated Hypotheses" if generated else "📝 Direct Hypotheses"
        self.print_subsection_banner(prefix, verbose)
        
        for i, hypothesis in enumerate(hypotheses, 1):
            # Create a styled hypothesis box
            hypothesis_box = Styling.box(f"{i}. {hypothesis}", self.palette.hypothesis)
            print(hypothesis_box)
            print()
    
    def print_approval_status_enhanced(self, status: str, details: Dict[str, Any] = None, 
                                     verbose: bool = True) -> None:
        """Print enhanced approval status with details."""
        if not verbose:
            return
        
        if status == "HYPOTHESIS VERIFIED":
            status_badge = self.status_formatter.format_status_badge("approved", "HYPOTHESIS VERIFIED")
            print(f"\n{status_badge}")
            
            if details:
                print(f"{self.palette.success}Details:{Colors.RESET}")
                for key, value in details.items():
                    print(f"  {self.palette.accent}{key}:{Colors.RESET} {self.palette.highlight}{value}{Colors.RESET}")
        elif status == "NOT APPLICABLE":
            status_badge = self.status_formatter.format_status_badge("warning", "NOT APPLICABLE")
            print(f"\n{status_badge}")
            
            if details:
                print(f"{self.palette.warning}Details:{Colors.RESET}")
                for key, value in details.items():
                    print(f"  {self.palette.accent}{key}:{Colors.RESET} {self.palette.highlight}{value}{Colors.RESET}")
        else:  # HYPOTHESIS NOT VERIFIED
            status_badge = self.status_formatter.format_status_badge("rejected", "HYPOTHESIS NOT VERIFIED")
            print(f"\n{status_badge}")
            
            if details:
                print(f"{self.palette.error}Details:{Colors.RESET}")
                for key, value in details.items():
                    print(f"  {self.palette.accent}{key}:{Colors.RESET} {self.palette.highlight}{value}{Colors.RESET}")
        
        print()
    
    def print_session_summary(self, verbose: bool = True) -> None:
        """Print session summary with statistics."""
        if not verbose:
            return
        
        session_duration = time.time() - self.start_time
        log_entries = len(self.session_log)
        
        summary_data = {
            "Session Duration": f"{session_duration:.2f} seconds",
            "Log Entries": str(log_entries),
            "Start Time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "End Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.print_section_banner("Session Summary", verbose)
        for key, value in summary_data.items():
            print(f"  {self.palette.accent}{key}:{Colors.RESET} {self.palette.highlight}{value}{Colors.RESET}")
        print()


# Create a global UI instance
ui = UI()
