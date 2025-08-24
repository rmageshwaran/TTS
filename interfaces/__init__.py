"""
User interface implementations for TTS Virtual Microphone.

This package contains interface implementations:
- CLI (Command Line Interface) - Primary interface
- API (REST API Interface) - Web service interface
- GUI (Graphical User Interface) - Desktop interface (stub)
"""

from .cli_interface import CLIInterface
from .api_interface import APIInterface
from .gui_interface import GUIInterface

__all__ = [
    "CLIInterface",
    "APIInterface", 
    "GUIInterface"
]