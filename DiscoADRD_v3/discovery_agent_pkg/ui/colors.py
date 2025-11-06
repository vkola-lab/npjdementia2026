"""
Color scheme and styling definitions for the Discovery Agent UI.

This module provides a comprehensive color palette and styling system
for production-level output formatting.
"""

from typing import Dict, Any
from enum import Enum

class ColorTheme(Enum):
    """Available color themes for the UI."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"

class Colors:
    """ANSI color codes and styling definitions."""
    
    # Reset code
    RESET = "\033[0m"
    
    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"
    
    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

class ColorPalette:
    """Production-level color palette for different UI elements."""
    
    def __init__(self, theme: ColorTheme = ColorTheme.DEFAULT):
        """Initialize color palette with specified theme."""
        self.theme = theme
        self._setup_palette()
    
    def _setup_palette(self):
        """Setup color palette based on theme."""
        if self.theme == ColorTheme.DEFAULT:
            self._setup_default_palette()
        elif self.theme == ColorTheme.DARK:
            self._setup_dark_palette()
        elif self.theme == ColorTheme.LIGHT:
            self._setup_light_palette()
        elif self.theme == ColorTheme.HIGH_CONTRAST:
            self._setup_high_contrast_palette()
    
    def _setup_default_palette(self):
        """Setup default color palette."""
        self.primary = Colors.BRIGHT_CYAN
        self.secondary = Colors.BRIGHT_BLUE
        self.success = Colors.BRIGHT_GREEN
        self.warning = Colors.BRIGHT_YELLOW
        self.error = Colors.BRIGHT_RED
        self.info = Colors.BRIGHT_BLUE
        self.muted = Colors.BRIGHT_BLACK
        self.highlight = Colors.BRIGHT_WHITE
        self.accent = Colors.BRIGHT_MAGENTA
        
        # Specific UI elements
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
        
        # Backgrounds
        self.bg_primary = Colors.BG_BLUE
        self.bg_success = Colors.BG_GREEN
        self.bg_warning = Colors.BG_YELLOW
        self.bg_error = Colors.BG_RED
    
    def _setup_dark_palette(self):
        """Setup dark theme color palette."""
        self.primary = Colors.CYAN
        self.secondary = Colors.BLUE
        self.success = Colors.GREEN
        self.warning = Colors.YELLOW
        self.error = Colors.RED
        self.info = Colors.BLUE
        self.muted = Colors.BLACK
        self.highlight = Colors.WHITE
        self.accent = Colors.MAGENTA
        
        # Specific UI elements
        self.banner = Colors.CYAN
        self.separator = Colors.BLACK
        self.progress = Colors.MAGENTA
        self.agent_scientist = Colors.GREEN
        self.agent_critic = Colors.RED
        self.agent_planner = Colors.BLUE
        self.hypothesis = Colors.WHITE
        self.code = Colors.BLACK
        self.table_header = Colors.BLUE
        self.table_row = Colors.WHITE
        self.status_approved = Colors.GREEN
        self.status_rejected = Colors.RED
        self.status_processing = Colors.YELLOW
        
        # Backgrounds
        self.bg_primary = Colors.BG_BLUE
        self.bg_success = Colors.BG_GREEN
        self.bg_warning = Colors.BG_YELLOW
        self.bg_error = Colors.BG_RED
    
    def _setup_light_palette(self):
        """Setup light theme color palette."""
        self.primary = Colors.BLUE
        self.secondary = Colors.CYAN
        self.success = Colors.GREEN
        self.warning = Colors.YELLOW
        self.error = Colors.RED
        self.info = Colors.BLUE
        self.muted = Colors.BLACK
        self.highlight = Colors.BLACK
        self.accent = Colors.MAGENTA
        
        # Specific UI elements
        self.banner = Colors.BLUE
        self.separator = Colors.BLACK
        self.progress = Colors.MAGENTA
        self.agent_scientist = Colors.GREEN
        self.agent_critic = Colors.RED
        self.agent_planner = Colors.BLUE
        self.hypothesis = Colors.BLACK
        self.code = Colors.BLACK
        self.table_header = Colors.BLUE
        self.table_row = Colors.BLACK
        self.status_approved = Colors.GREEN
        self.status_rejected = Colors.RED
        self.status_processing = Colors.YELLOW
        
        # Backgrounds
        self.bg_primary = Colors.BG_BLUE
        self.bg_success = Colors.BG_GREEN
        self.bg_warning = Colors.BG_YELLOW
        self.bg_error = Colors.BG_RED
    
    def _setup_high_contrast_palette(self):
        """Setup high contrast color palette."""
        self.primary = Colors.BRIGHT_WHITE
        self.secondary = Colors.BRIGHT_CYAN
        self.success = Colors.BRIGHT_GREEN
        self.warning = Colors.BRIGHT_YELLOW
        self.error = Colors.BRIGHT_RED
        self.info = Colors.BRIGHT_CYAN
        self.muted = Colors.WHITE
        self.highlight = Colors.BRIGHT_WHITE
        self.accent = Colors.BRIGHT_MAGENTA
        
        # Specific UI elements
        self.banner = Colors.BRIGHT_WHITE
        self.separator = Colors.WHITE
        self.progress = Colors.BRIGHT_MAGENTA
        self.agent_scientist = Colors.BRIGHT_GREEN
        self.agent_critic = Colors.BRIGHT_RED
        self.agent_planner = Colors.BRIGHT_CYAN
        self.hypothesis = Colors.BRIGHT_WHITE
        self.code = Colors.WHITE
        self.table_header = Colors.BRIGHT_CYAN
        self.table_row = Colors.WHITE
        self.status_approved = Colors.BRIGHT_GREEN
        self.status_rejected = Colors.BRIGHT_RED
        self.status_processing = Colors.BRIGHT_YELLOW
        
        # Backgrounds
        self.bg_primary = Colors.BG_BLUE
        self.bg_success = Colors.BG_GREEN
        self.bg_warning = Colors.BG_YELLOW
        self.bg_error = Colors.BG_RED

class Styling:
    """Text styling and formatting utilities."""
    
    @staticmethod
    def bold(text: str) -> str:
        """Make text bold."""
        return f"{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def italic(text: str) -> str:
        """Make text italic."""
        return f"{Colors.ITALIC}{text}{Colors.RESET}"
    
    @staticmethod
    def underline(text: str) -> str:
        """Underline text."""
        return f"{Colors.UNDERLINE}{text}{Colors.RESET}"
    
    @staticmethod
    def dim(text: str) -> str:
        """Make text dim."""
        return f"{Colors.DIM}{text}{Colors.RESET}"
    
    @staticmethod
    def highlight(text: str, color: str) -> str:
        """Highlight text with specified color."""
        return f"{color}{text}{Colors.RESET}"
    
    @staticmethod
    def box(text: str, color: str = Colors.BRIGHT_CYAN) -> str:
        """Create a box around text."""
        lines = text.split('\n')
        max_width = max(len(line) for line in lines)
        
        box_lines = [
            f"{color}┌{'─' * (max_width + 2)}┐{Colors.RESET}",
            *[f"{color}│ {line.ljust(max_width)} │{Colors.RESET}" for line in lines],
            f"{color}└{'─' * (max_width + 2)}┘{Colors.RESET}"
        ]
        
        return '\n'.join(box_lines)
    
    @staticmethod
    def progress_bar(current: int, total: int, width: int = 50, 
                    fill_char: str = "█", empty_char: str = "░") -> str:
        """Create a progress bar."""
        if total == 0:
            return f"{empty_char * width} 0%"
        
        progress = current / total
        filled_width = int(width * progress)
        bar = fill_char * filled_width + empty_char * (width - filled_width)
        percentage = int(progress * 100)
        
        return f"[{bar}] {percentage}%"
    
    @staticmethod
    def status_indicator(status: str, color: str) -> str:
        """Create a status indicator."""
        status_symbols = {
            "success": "●",
            "error": "●", 
            "warning": "●",
            "info": "●",
            "processing": "◐",
            "pending": "○",
            "approved": "✓",
            "rejected": "✗"
        }
        
        symbol = status_symbols.get(status.lower(), "●")
        return f"{color}{symbol}{Colors.RESET}"

# Global color palette instance
default_palette = ColorPalette(ColorTheme.DEFAULT)
