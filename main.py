#!/usr/bin/env python3
"""
TTS Virtual Microphone - Main Application Entry Point

This is the main entry point for the TTS Virtual Microphone application.
It handles command-line arguments and coordinates the various components.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config_manager import ConfigManager
from utils.logger import setup_logging
from utils.error_handler import handle_exception
from interfaces.cli_interface import CLIInterface
from interfaces.api_interface import APIInterface
from interfaces.gui_interface import GUIInterface


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TTS Virtual Microphone - Convert text to speech and route to virtual microphone"
    )
    
    # Text input options
    parser.add_argument("--text", "-t", type=str, help="Text to convert to speech")
    parser.add_argument("--file", "-f", type=str, help="File containing text to convert")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    
    # Interface options
    parser.add_argument("--api", action="store_true", help="Start REST API server")
    parser.add_argument("--gui", action="store_true", help="Start GUI interface")
    parser.add_argument("--port", type=int, default=8080, help="API server port (default: 8080)")
    
    # Voice settings
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (default: 0)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Voice temperature (default: 1.0)")
    
    # System options
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Testing and diagnostics
    parser.add_argument("--test", action="store_true", help="Run system tests")
    parser.add_argument("--test-tts", action="store_true", help="Test TTS engine")
    parser.add_argument("--test-audio", action="store_true", help="Test virtual audio")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.load()
        # Use raw config dictionary to preserve all YAML sections (like tts.sesame_csm)
        raw_config = config_manager.raw_config
        
        # Setup logging
        log_level = "DEBUG" if args.debug else config.application.log_level
        setup_logging(level=log_level, verbose=args.verbose)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting TTS Virtual Microphone Application")
        
        # Choose interface based on arguments
        if args.api:
            # Start REST API server
            logger.info("Starting REST API interface")
            interface = APIInterface(raw_config)
            interface.start(port=args.port)
            
        elif args.gui:
            # Start GUI interface
            logger.info("Starting GUI interface")
            interface = GUIInterface(raw_config)
            interface.start()
            
        elif args.test:
            # Run system tests
            logger.info("Running system tests")
            from tests.system_test import run_system_tests
            run_system_tests(raw_config)
            
        elif args.test_tts:
            # Test TTS engine
            logger.info("Testing TTS engine")
            from tests.tts_test import test_tts_engine
            test_tts_engine(raw_config)
            
        elif args.test_audio:
            # Test virtual audio
            logger.info("Testing virtual audio")
            from tests.audio_test import test_virtual_audio
            test_virtual_audio(raw_config)
            
        elif args.list_devices:
            # List audio devices
            from utils.device_manager import DeviceManager
            device_manager = DeviceManager()
            device_manager.list_devices()
            
        else:
            # Default to CLI interface
            logger.info("Starting CLI interface")
            interface = CLIInterface(raw_config)
            
            if args.interactive:
                interface.start_interactive()
            elif args.text:
                interface.process_text(args.text, speaker_id=args.speaker, temperature=args.temperature)
            elif args.file:
                interface.process_file(args.file, speaker_id=args.speaker, temperature=args.temperature)
            else:
                interface.show_help()
                
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()