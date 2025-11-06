"""
Advanced formatting and display components for the Discovery Agent UI.

This module provides production-level formatting for various UI elements
including tables, reports, banners, and structured content.
"""

import textwrap
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from .colors import Colors, Styling, default_palette

class TableFormatter:
    """Advanced table formatting with multiple styles."""
    
    def __init__(self, palette=None):
        """Initialize table formatter."""
        self.palette = palette or default_palette
    
    def format_table(self, data: List[Dict[str, Any]], 
                    headers: Optional[List[str]] = None,
                    style: str = "default",
                    max_width: int = 120) -> str:
        """Format data as a table with various styles."""
        if not data:
            return ""
        
        if headers is None:
            headers = list(data[0].keys())
        
        if style == "minimal":
            return self._format_minimal_table(data, headers, max_width)
        elif style == "boxed":
            return self._format_boxed_table(data, headers, max_width)
        elif style == "grid":
            return self._format_grid_table(data, headers, max_width)
        else:
            return self._format_default_table(data, headers, max_width)
    
    def _format_default_table(self, data: List[Dict[str, Any]], 
                             headers: List[str], max_width: int) -> str:
        """Format table with default style."""
        # Calculate column widths
        col_widths = self._calculate_column_widths(data, headers, max_width)
        
        # Create header
        header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
        header_line = f"{self.palette.table_header}{header_row}{Colors.RESET}"
        
        # Create separator
        separator = "-+-".join("-" * col_widths[header] for header in headers)
        separator_line = f"{self.palette.separator}{separator}{Colors.RESET}"
        
        # Create data rows
        data_rows = []
        for row in data:
            row_data = []
            for header in headers:
                value = str(row.get(header, ""))
                # Truncate if too long
                if len(value) > col_widths[header]:
                    value = value[:col_widths[header]-3] + "..."
                row_data.append(value.ljust(col_widths[header]))
            data_rows.append(f"{self.palette.table_row}{' | '.join(row_data)}{Colors.RESET}")
        
        return "\n".join([header_line, separator_line] + data_rows)
    
    def _format_minimal_table(self, data: List[Dict[str, Any]], 
                             headers: List[str], max_width: int) -> str:
        """Format table with minimal style."""
        col_widths = self._calculate_column_widths(data, headers, max_width)
        
        lines = []
        for row in data:
            row_data = []
            for header in headers:
                value = str(row.get(header, ""))
                if len(value) > col_widths[header]:
                    value = value[:col_widths[header]-3] + "..."
                row_data.append(value.ljust(col_widths[header]))
            lines.append("  ".join(row_data))
        
        return "\n".join(lines)
    
    def _format_boxed_table(self, data: List[Dict[str, Any]], 
                           headers: List[str], max_width: int) -> str:
        """Format table with boxed style."""
        col_widths = self._calculate_column_widths(data, headers, max_width)
        
        # Top border
        top_border = "┌" + "┬".join("─" * (col_widths[header] + 2) for header in headers) + "┐"
        
        # Header row
        header_row = "│ " + " │ ".join(header.center(col_widths[header]) for header in headers) + " │"
        
        # Middle border
        middle_border = "├" + "┼".join("─" * (col_widths[header] + 2) for header in headers) + "┤"
        
        # Data rows
        data_rows = []
        for row in data:
            row_data = []
            for header in headers:
                value = str(row.get(header, ""))
                if len(value) > col_widths[header]:
                    value = value[:col_widths[header]-3] + "..."
                row_data.append(value.ljust(col_widths[header]))
            data_rows.append("│ " + " │ ".join(row_data) + " │")
        
        # Bottom border
        bottom_border = "└" + "┴".join("─" * (col_widths[header] + 2) for header in headers) + "┘"
        
        all_lines = [top_border, header_row, middle_border] + data_rows + [bottom_border]
        return "\n".join(all_lines)
    
    def _format_grid_table(self, data: List[Dict[str, Any]], 
                          headers: List[str], max_width: int) -> str:
        """Format table with grid style."""
        col_widths = self._calculate_column_widths(data, headers, max_width)
        
        # Create grid
        lines = []
        
        # Header
        header_row = "┌" + "┬".join("─" * (col_widths[header] + 2) for header in headers) + "┐"
        lines.append(header_row)
        
        header_data = "│ " + " │ ".join(header.center(col_widths[header]) for header in headers) + " │"
        lines.append(header_data)
        
        # Data rows
        for i, row in enumerate(data):
            # Row separator
            if i > 0:
                separator = "├" + "┼".join("─" * (col_widths[header] + 2) for header in headers) + "┤"
                lines.append(separator)
            
            # Data row
            row_data = []
            for header in headers:
                value = str(row.get(header, ""))
                if len(value) > col_widths[header]:
                    value = value[:col_widths[header]-3] + "..."
                row_data.append(value.ljust(col_widths[header]))
            lines.append("│ " + " │ ".join(row_data) + " │")
        
        # Bottom border
        bottom_border = "└" + "┴".join("─" * (col_widths[header] + 2) for header in headers) + "┘"
        lines.append(bottom_border)
        
        return "\n".join(lines)
    
    def _calculate_column_widths(self, data: List[Dict[str, Any]], 
                                headers: List[str], max_width: int) -> Dict[str, int]:
        """Calculate optimal column widths."""
        col_widths = {}
        
        # Start with header widths
        for header in headers:
            col_widths[header] = len(header)
        
        # Add data widths
        for row in data:
            for header in headers:
                value = str(row.get(header, ""))
                col_widths[header] = max(col_widths[header], len(value))
        
        # Adjust for max width
        total_width = sum(col_widths.values()) + (len(headers) - 1) * 3  # 3 for separators
        
        if total_width > max_width:
            # Proportionally reduce column widths
            ratio = max_width / total_width
            for header in headers:
                col_widths[header] = max(8, int(col_widths[header] * ratio))
        
        return col_widths

class ReportFormatter:
    """Advanced report formatting with sections and styling."""
    
    def __init__(self, palette=None):
        """Initialize report formatter."""
        self.palette = palette or default_palette
    
    def format_experiment_report(self, experiment_data: Dict[str, Any]) -> str:
        """Format comprehensive experiment report."""
        sections = []
        
        # Header
        sections.append(self._format_header("Experiment Report", experiment_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
        
        # Summary
        if "summary" in experiment_data:
            sections.append(self._format_section("Summary", experiment_data["summary"]))
        
        # Configuration
        if "configuration" in experiment_data:
            sections.append(self._format_section("Configuration", experiment_data["configuration"]))
        
        # Results
        if "results" in experiment_data:
            sections.append(self._format_section("Results", experiment_data["results"]))
        
        # Statistics
        if "statistics" in experiment_data:
            sections.append(self._format_section("Statistics", experiment_data["statistics"]))
        
        # Timeline
        if "timeline" in experiment_data:
            sections.append(self._format_section("Timeline", experiment_data["timeline"]))
        
        return "\n\n".join(sections)
    
    def _format_header(self, title: str, subtitle: str = "") -> str:
        """Format report header."""
        width = 80
        title_line = f" {title} ".center(width, "=")
        header = f"{self.palette.banner}{title_line}{Colors.RESET}"
        
        if subtitle:
            subtitle_line = f" {subtitle} ".center(width, " ")
            header += f"\n{self.palette.muted}{subtitle_line}{Colors.RESET}"
        
        return header
    
    def _format_section(self, title: str, content: Union[str, Dict, List]) -> str:
        """Format a report section."""
        section_header = f"\n{self.palette.secondary}{title.upper()}{Colors.RESET}\n{'-' * len(title)}"
        
        if isinstance(content, str):
            section_content = self._format_text_content(content)
        elif isinstance(content, dict):
            section_content = self._format_dict_content(content)
        elif isinstance(content, list):
            section_content = self._format_list_content(content)
        else:
            section_content = str(content)
        
        return f"{section_header}\n{section_content}"
    
    def _format_text_content(self, text: str) -> str:
        """Format text content with proper wrapping."""
        wrapped_text = textwrap.fill(text, width=78, initial_indent="  ", subsequent_indent="  ")
        return f"{self.palette.highlight}{wrapped_text}{Colors.RESET}"
    
    def _format_dict_content(self, data: Dict[str, Any]) -> str:
        """Format dictionary content as key-value pairs."""
        lines = []
        for key, value in data.items():
            key_formatted = f"{self.palette.accent}{key}:{Colors.RESET}"
            value_formatted = f"{self.palette.highlight}{value}{Colors.RESET}"
            lines.append(f"  {key_formatted} {value_formatted}")
        return "\n".join(lines)
    
    def _format_list_content(self, data: List[Any]) -> str:
        """Format list content with bullets."""
        lines = []
        for item in data:
            item_formatted = f"{self.palette.highlight}{item}{Colors.RESET}"
            lines.append(f"  • {item_formatted}")
        return "\n".join(lines)

class BannerFormatter:
    """Advanced banner and header formatting."""
    
    def __init__(self, palette=None):
        """Initialize banner formatter."""
        self.palette = palette or default_palette
    
    def format_main_banner(self, title: str = "Discovery Agent", 
                          subtitle: str = "AI-Powered Scientific Discovery System",
                          version: str = "1.0.0") -> str:
        """Format main application banner."""
        width = 80
        
        # Top border
        top_border = f"{self.palette.banner}╔{'═' * (width - 2)}╗{Colors.RESET}"
        
        # Title line
        title_line = f"{self.palette.banner}║{title.center(width - 2)}║{Colors.RESET}"
        
        # Subtitle line
        subtitle_line = f"{self.palette.banner}║{subtitle.center(width - 2)}║{Colors.RESET}"
        
        # Empty line
        empty_line = f"{self.palette.banner}║{' ' * (width - 2)}║{Colors.RESET}"
        
        # Features line
        features = "Multi-Agent System • Hypothesis Testing • Data Analysis"
        features_line = f"{self.palette.banner}║{features.center(width - 2)}║{Colors.RESET}"
        
        # Version line
        version_text = f"Version {version}"
        version_line = f"{self.palette.banner}║{version_text.center(width - 2)}║{Colors.RESET}"
        
        # Bottom border
        bottom_border = f"{self.palette.banner}╚{'═' * (width - 2)}╝{Colors.RESET}"
        
        return "\n".join([
            top_border,
            title_line,
            subtitle_line,
            empty_line,
            features_line,
            version_line,
            bottom_border
        ])
    
    def format_section_banner(self, title: str, width: int = 60) -> str:
        """Format section banner."""
        title_line = f" {title} ".center(width, "=")
        return f"\n{self.palette.primary}{title_line}{Colors.RESET}\n"
    
    def format_subsection_banner(self, title: str, width: int = 40) -> str:
        """Format subsection banner."""
        title_line = f" {title} ".center(width, "-")
        return f"\n{self.palette.secondary}{title_line}{Colors.RESET}\n"

class CodeFormatter:
    """Code formatting and syntax highlighting."""
    
    def __init__(self, palette=None):
        """Initialize code formatter."""
        self.palette = palette or default_palette
    
    def format_code_block(self, code: str, language: str = "python") -> str:
        """Format code block with syntax highlighting."""
        lines = code.split('\n')
        
        # Header
        header = f"{self.palette.code}```{language}{Colors.RESET}"
        
        # Code lines
        code_lines = []
        for line in lines:
            # Simple syntax highlighting for Python
            if language == "python":
                highlighted_line = self._highlight_python_line(line)
            else:
                highlighted_line = line
            
            code_lines.append(f"{self.palette.code}{highlighted_line}{Colors.RESET}")
        
        # Footer
        footer = f"{self.palette.code}```{Colors.RESET}"
        
        return "\n".join([header] + code_lines + [footer])
    
    def _highlight_python_line(self, line: str) -> str:
        """Simple Python syntax highlighting."""
        # Keywords
        keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'import', 'from', 'return', 'yield', 'lambda', 'with', 'as']
        
        # String highlighting
        if line.strip().startswith('#'):
            return f"{self.palette.muted}{line}{Colors.RESET}"
        
        # Simple keyword highlighting
        for keyword in keywords:
            if f" {keyword} " in f" {line} " or line.strip().startswith(keyword):
                line = line.replace(keyword, f"{self.palette.accent}{keyword}{Colors.RESET}")
        
        return line
    
    def format_output_block(self, output: str, title: str = "Output") -> str:
        """Format output block."""
        header = f"{self.palette.info}📤 {title}:{Colors.RESET}"
        output_lines = [f"{self.palette.code}{line}{Colors.RESET}" for line in output.split('\n')]
        
        return "\n".join([header] + output_lines)

class StatusFormatter:
    """Status and indicator formatting."""
    
    def __init__(self, palette=None):
        """Initialize status formatter."""
        self.palette = palette or default_palette
    
    def format_status_badge(self, status: str, text: str = "") -> str:
        """Format status badge."""
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
        
        status_symbols = {
            "success": "✓",
            "error": "✗",
            "warning": "⚠",
            "info": "ℹ",
            "processing": "◐",
            "pending": "○",
            "approved": "✓",
            "rejected": "✗"
        }
        
        color = status_colors.get(status.lower(), self.palette.muted)
        symbol = status_symbols.get(status.lower(), "●")
        
        if text:
            return f"{color}[{symbol}] {text}{Colors.RESET}"
        else:
            return f"{color}{symbol}{Colors.RESET}"
    
    def format_progress_indicator(self, current: int, total: int, 
                                 label: str = "", width: int = 20) -> str:
        """Format progress indicator."""
        if total == 0:
            return f"{self.palette.muted}[{'░' * width}] 0%{Colors.RESET}"
        
        progress = current / total
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = int(progress * 100)
        
        if label:
            return f"{label}: {self.palette.progress}[{bar}] {percentage}%{Colors.RESET}"
        else:
            return f"{self.palette.progress}[{bar}] {percentage}%{Colors.RESET}"
