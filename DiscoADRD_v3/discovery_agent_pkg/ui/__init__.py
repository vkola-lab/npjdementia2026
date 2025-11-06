"""
Enhanced UI module for the Discovery Agent.

This module provides production-level user interface components including:
- Advanced message printing with colors and emojis
- Progress tracking and status displays
- Enhanced formatting and styling
- User input handling
- Agent response display
- Session logging and reporting
"""

from .interface import ui
from .colors import Colors, Styling, default_palette, ColorTheme, ColorPalette
from .progress import ProgressTracker, ProgressDisplay, StatusDisplay, ExperimentProgress
from .formatting import TableFormatter, ReportFormatter, BannerFormatter, CodeFormatter, StatusFormatter

__all__ = [
    'ui',
    'Colors', 'Styling', 'default_palette', 'ColorTheme', 'ColorPalette',
    'ProgressTracker', 'ProgressDisplay', 'StatusDisplay', 'ExperimentProgress',
    'TableFormatter', 'ReportFormatter', 'BannerFormatter', 'CodeFormatter', 'StatusFormatter'
]
