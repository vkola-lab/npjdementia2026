"""
Simple user interface for the Discovery Agent.
"""

import emoji
import time
from colorama import Fore, Back, Style, init
from .utils import create_log_file, remove_ansi_color_codes

# Initialize colorama for cross-platform color support
init(autoreset=True)


class UI:
    """Simple UI class."""
    
    def __init__(self):
        """Initialize UI with logging setup."""
        self.log_file = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup simple file logging."""
        try:
            # Create log file with timestamp
            log_file_path = create_log_file("session_log")
            self.log_file = open(log_file_path, "w", encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not setup logging: {e}")
            self.log_file = None
    
    def _write_to_log(self, message: str) -> None:
        """Write message to log file."""
        if self.log_file:
            try:
                # Remove color codes for clean log file
                clean_message = self._remove_color_codes(message)
                self.log_file.write(clean_message + "\n")
                self.log_file.flush()
            except Exception:
                pass  # Ignore logging errors
    
    def _remove_color_codes(self, text: str) -> str:
        """Remove ANSI color codes from text."""
        return remove_ansi_color_codes(text)
    
    def print_banner(self) -> None:
        """Print application banner."""
        banner = f"""
{Fore.CYAN}{Style.BRIGHT}{"="*70}
🧠           Discovery Agent    🔬
    AI-Powered Scientific Discovery    🚀
{Style.BRIGHT}{"="*70}{Style.RESET_ALL}\n"""
        print(banner)
        self._write_to_log(banner)
    
    def get_user_input(self, prompt: str = "Enter your hypothesis or question") -> str:
        """Get user input."""
        colored_prompt = f"{Fore.YELLOW}{Style.BRIGHT}🤔 {prompt}{Style.RESET_ALL}: "
        user_input = input(colored_prompt).strip()
        # Log the prompt and user input
        self._write_to_log(f"🤔 {prompt}: {user_input}")
        return user_input
    
    def get_hypothesis_mode_choice(self) -> bool:
        """Ask user whether to generate hypotheses or use input directly."""
        prompt = f"{Fore.CYAN}{Style.BRIGHT}🔬 Hypothesis Generation Mode{Style.RESET_ALL}\n"
        prompt += f"{Fore.WHITE}Choose how to proceed:{Style.RESET_ALL}\n"
        prompt += f"{Fore.GREEN}  (1){Style.RESET_ALL} Generate multiple hypotheses from your input\n"
        prompt += f"{Fore.BLUE}  (2){Style.RESET_ALL} Use your input directly as a single hypothesis\n"
        prompt += f"{Fore.YELLOW}{Style.BRIGHT}Enter your choice (1 or 2){Style.RESET_ALL}: "
        
        while True:
            choice = input(prompt).strip()
            self._write_to_log(f"🔬 Hypothesis Generation Mode Choice: {choice}")
            
            if choice == "1":
                return True  # Generate hypotheses
            elif choice == "2":
                return False  # Use input directly
            else:
                print(f"{Fore.RED}❌ Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")
                self._write_to_log("❌ Invalid choice entered, asking again")
    
    def get_preprocessing_mode_choice(self) -> str:
        """Ask user to choose between regular flow and manual preprocessing mode."""
        prompt = f"{Fore.CYAN}{Style.BRIGHT}🔧 Preprocessing Mode Selection{Style.RESET_ALL}\n"
        prompt += f"{Fore.WHITE}Choose how to proceed:{Style.RESET_ALL}\n"
        prompt += f"{Fore.GREEN}  (1){Style.RESET_ALL} Regular mode - Enter hypothesis and let the system handle data preprocessing\n"
        prompt += f"{Fore.BLUE}  (2){Style.RESET_ALL} Manual mode - Provide folder path, dictionary, and hypothesis separately\n"
        prompt += f"{Fore.YELLOW}{Style.BRIGHT}Enter your choice (1 or 2){Style.RESET_ALL}: "
        
        while True:
            choice = input(prompt).strip()
            self._write_to_log(f"🔧 Preprocessing Mode Choice: {choice}")
            
            if choice == "1":
                return "regular"  # Regular mode
            elif choice == "2":
                return "manual"  # Manual preprocessing mode
            else:
                print(f"{Fore.RED}❌ Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")
                self._write_to_log("❌ Invalid choice entered, asking again")
    
    def get_folder_path(self, prompt: str = "Enter the path to the folder containing CSV files") -> str:
        """Get folder path from user with validation."""
        import os
        
        while True:
            colored_prompt = f"{Fore.YELLOW}{Style.BRIGHT}📁 {prompt}{Style.RESET_ALL}: "
            folder_path = input(colored_prompt).strip()
            self._write_to_log(f"📁 {prompt}: {folder_path}")
            
            if not folder_path:
                print(f"{Fore.RED}❌ Please enter a valid folder path.{Style.RESET_ALL}")
                continue
            
            # Expand user path and make absolute
            folder_path = os.path.expanduser(folder_path)
            folder_path = os.path.abspath(folder_path)
            
            if not os.path.exists(folder_path):
                print(f"{Fore.RED}❌ Folder does not exist: {folder_path}{Style.RESET_ALL}")
                continue
            
            if not os.path.isdir(folder_path):
                print(f"{Fore.RED}❌ Path is not a directory: {folder_path}{Style.RESET_ALL}")
                continue
            
            # Check if folder contains CSV and JSON files
            csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
            json_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
            
            if not csv_files:
                print(f"{Fore.RED}❌ No CSV files found in the folder: {folder_path}{Style.RESET_ALL}")
                continue
            
            if not json_files:
                print(f"{Fore.RED}❌ No JSON files found in the folder: {folder_path}{Style.RESET_ALL}")
                continue
            
            print(f"{Fore.GREEN}✅ Found {len(csv_files)} CSV file(s) and {len(json_files)} JSON file(s) in folder: {folder_path}{Style.RESET_ALL}")
            self._write_to_log(f"✅ Found {len(csv_files)} CSV file(s) and {len(json_files)} JSON file(s) in folder: {folder_path}")
            return folder_path
    
    
    def print_info(self, message: str, verbose: bool = True) -> None:
        """Print info message."""
        formatted_message = f"{Fore.BLUE}{Style.BRIGHT}ℹ️ Info{Style.RESET_ALL}: {message}"
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def print_success(self, message: str, verbose: bool = True) -> None:
        """Print success message."""
        formatted_message = f"{Fore.GREEN}{Style.BRIGHT}✅ Success{Style.RESET_ALL}: {message}"
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def print_warning(self, message: str, verbose: bool = True) -> None:
        """Print warning message."""
        formatted_message = f"{Fore.YELLOW}{Style.BRIGHT}⚠️ Warning{Style.RESET_ALL}: {message}"
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def print_error(self, error: str, context: str = "", verbose: bool = True) -> None:
        """Print error message."""
        if context:
            formatted_message = f"{Fore.RED}{Style.BRIGHT}❌ Error in {context}{Style.RESET_ALL}: {error}"
        else:
            formatted_message = f"{Fore.RED}{Style.BRIGHT}❌ Error{Style.RESET_ALL}: {error}"
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def print_agent_response(self, agent_name: str, response: str, code_output: str = "", verbose: bool = True) -> None:
        """Print agent response."""
        # Choose emoji and color based on agent
        if "Scientist" in agent_name:
            agent_emoji = "👨‍🔬"
            agent_color = Fore.CYAN
        elif "Critic" in agent_name:
            agent_emoji = "🕵️"
            agent_color = Fore.MAGENTA
        else:
            agent_emoji = "🤖"
            agent_color = Fore.WHITE
        
        agent_header = f"\n{agent_color}{Style.BRIGHT}{agent_emoji} --- {agent_name} ---{Style.RESET_ALL}"
        if verbose:
            print(agent_header)
            print(response)
        self._write_to_log(agent_header)
        self._write_to_log(response)
        
        if code_output:
            code_output_formatted = f"\n{Fore.GREEN}{Style.BRIGHT}💻 Execution Output{Style.RESET_ALL}:\n{code_output}"
            if verbose:
                print(code_output_formatted)
            self._write_to_log(code_output_formatted)
    
    def print_approval_status(self, approved: bool, verbose: bool = True) -> None:
        """Print approval status."""
        if approved:
            formatted_message = f"\n{Fore.GREEN}{Style.BRIGHT}🎉 Critic: RESULTS APPROVED{Style.RESET_ALL}. 🏁 Conversation terminated."
        else:
            formatted_message = f"\n{Fore.RED}{Style.BRIGHT}🚫 Critic: RESULTS NOT APPROVED{Style.RESET_ALL}. 💬 Please provide feedback."
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def print_separator(self, char: str = "─", length: int = 60, verbose: bool = True) -> None:
        """Print a visual separator line."""
        separator = f"{Fore.CYAN}{char * length}{Style.RESET_ALL}"
        if verbose:
            print(separator)
        self._write_to_log(separator)
    
    def print_loading(self, message: str = "Processing...", verbose: bool = True) -> None:
        """Print a loading message with spinner."""
        formatted_message = f"{Fore.YELLOW}{Style.BRIGHT}⏳ {message}{Style.RESET_ALL}"
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def print_hypothesis_list(self, hypotheses: list, generated: bool = True, verbose: bool = True) -> None:
        """Print a formatted list of hypotheses."""
        if generated:
            header = f"\n{Fore.CYAN}{Style.BRIGHT}💡 Generated Hypotheses:{Style.RESET_ALL}"
        else:
            header = f"\n{Fore.CYAN}{Style.BRIGHT}📝 Using Input as Hypothesis:{Style.RESET_ALL}"
        if verbose:
            print(header)
        self._write_to_log(header)
        for idx, hypothesis in enumerate(hypotheses, 1):
            hypothesis_line = f"{Fore.WHITE}{Style.BRIGHT}  {idx}.{Style.RESET_ALL} {hypothesis}"
            if verbose:
                print(hypothesis_line)
            self._write_to_log(hypothesis_line)
        if verbose:
            print()
        self._write_to_log("")
    
    def print_progress(self, current: int, total: int, item: str = "hypothesis", verbose: bool = True) -> None:
        """Print progress information."""
        percentage = (current / total) * 100 if total > 0 else 0
        formatted_message = f"{Fore.BLUE}{Style.BRIGHT}📊 Progress: {current}/{total} {item}s ({percentage:.1f}%){Style.RESET_ALL}"
        if verbose:
            print(formatted_message)
        self._write_to_log(formatted_message)
    
    def close_log(self) -> None:
        """Close the log file properly."""
        if self.log_file:
            try:
                self.log_file.close()
                self.log_file = None
            except Exception:
                pass


# Global UI instance
ui = UI()
