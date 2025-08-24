#!/usr/bin/env python3
"""
Command Line Interface for TTS Virtual Microphone

Provides command-line interaction for text-to-speech processing
and virtual microphone routing functionality.
"""

import sys
import os
import logging
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Import core components
from core.text_processor import TextProcessor
from core.tts_engine import TTSEngine
from core.audio_processor import AudioProcessor
from core.virtual_audio import VirtualAudioInterface
from utils.error_handler import handle_exception

logger = logging.getLogger(__name__)


class CLIInterface:
    """
    Command Line Interface for TTS Virtual Microphone.
    
    Provides basic CLI functionality for processing text input,
    managing TTS settings, and routing audio to virtual microphone.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CLI interface.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        
        # Initialize core components
        try:
            # Convert Config object to dictionary for component initialization
            if hasattr(config, '__dict__'):
                config_dict = self._config_to_dict(config)
            else:
                config_dict = config
            
            self.text_processor = TextProcessor(config_dict)
            self.tts_engine = TTSEngine(config_dict)
            self.audio_processor = AudioProcessor(config_dict)
            self.virtual_audio = VirtualAudioInterface(config_dict)
            
            logger.info("CLI interface initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI components: {e}")
            raise
    
    def _config_to_dict(self, config) -> Dict[str, Any]:
        """
        Convert Config object to dictionary format expected by components.
        
        Args:
            config: Config dataclass object
            
        Returns:
            Dictionary representation of config
        """
        from dataclasses import asdict
        
        try:
            if hasattr(config, '__dataclass_fields__'):
                return asdict(config)
            else:
                # Already a dictionary or dictionary-like
                return config
        except Exception as e:
            logger.warning(f"Failed to convert config to dict: {e}")
            # Return basic config structure
            return {
                'application': {'log_level': 'INFO'},
                'tts': {'default_engine': 'demo_tts', 'context_length': 4},
                'audio': {'sample_rate': 24000, 'channels': 1, 'format': 'wav'},
                'virtual_audio': {'device_name': 'CABLE Input', 'auto_detect': True}
            }
    
    def show_help(self):
        """Display help information and usage examples."""
        help_text = """
🎤 TTS Virtual Microphone - Command Line Interface

USAGE:
    python main.py [OPTIONS]

OPTIONS:
    --text, -t TEXT         Convert text to speech
    --file, -f FILE         Process text from file
    --interactive, -i       Start interactive mode
    
    --speaker ID            Speaker ID (default: 0)
    --temperature VALUE     Voice temperature (default: 1.0)
    
    --api                   Start REST API server
    --gui                   Start GUI interface
    --port PORT             API server port (default: 8080)
    
    --config, -c FILE       Configuration file path
    --debug, -d             Enable debug logging
    --verbose, -v           Verbose output
    
    --test                  Run system tests
    --test-tts              Test TTS engine
    --test-audio            Test virtual audio
    --list-devices          List available audio devices
    
    --help, -h              Show this help message

EXAMPLES:
    # Convert text to speech
    python main.py --text "Hello, world!"
    
    # Process text file
    python main.py --file input.txt
    
    # Interactive mode
    python main.py --interactive
    
    # Use specific speaker and settings
    python main.py --text "Hello" --speaker 1 --temperature 0.8
    
    # Start API server
    python main.py --api --port 8080
    
    # Test system components
    python main.py --test-tts
    python main.py --test-audio

CONFIGURATION:
    Configuration is loaded from:
    1. Command line --config option
    2. config/config.yaml (default)
    
    See docs/README.md for detailed configuration options.

VIRTUAL AUDIO:
    This application routes TTS audio to virtual microphone devices.
    Install VB-Cable for best results: https://vb-audio.com/Cable/
    
    Supported virtual audio:
    - VB-Audio Virtual Cable
    - Windows default audio devices
    - File output (fallback)

For more information, visit: https://github.com/your-repo/tts-virtual-microphone
        """
        print(help_text)
    
    def process_text(self, text: str, speaker_id: int = 0, temperature: float = 1.0) -> bool:
        """
        Process a single text input through the TTS pipeline.
        
        Args:
            text: Text to convert to speech
            speaker_id: Speaker ID for voice selection
            temperature: Voice temperature setting
            
        Returns:
            bool: True if processing successful
        """
        try:
            print(f"Processing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Step 1: Text Processing
            print("   Processing text...")
            processed_result = self.text_processor.process(text)
            
            if hasattr(processed_result, 'processed_text'):
                processed_text = processed_result.processed_text
            else:
                processed_text = str(processed_result)
            
            print(f"   Text processed: {len(processed_text)} characters")
            
            # Step 2: TTS Generation
            print("   Generating speech...")
            tts_result = self.tts_engine.generate_speech(
                processed_text, 
                speaker_id=speaker_id
            )
            
            if not tts_result.success:
                print(f"   TTS generation failed: {tts_result.error_message}")
                return False
            
            print(f"   Speech generated: {len(tts_result.audio_data)} bytes")
            
            # Step 3: Audio Processing
            print("   Processing audio...")
            audio_result = self.audio_processor.process(tts_result.audio_data)
            
            if not audio_result.success:
                print(f"   Audio processing failed: {audio_result.error_message}")
                return False
            
            print(f"   Audio processed: {len(audio_result.audio_data)} bytes")
            
            # Step 3.5: Save audio file to output directory
            self._save_audio_file(audio_result.audio_data, processed_text)
            
            # Step 4: Virtual Audio Routing
            print("   Routing to virtual microphone...")
            
            # Try VB-Cable first if available
            if self.virtual_audio.is_vb_cable_available():
                routing_result = self.virtual_audio.route_via_vb_cable(audio_result.audio_data)
                if routing_result.success:
                    print(f"   Routed via VB-Cable: {routing_result.device_used}")
                    print(f"      Latency: {routing_result.latency:.2f}ms")
                else:
                    print(f"   VB-Cable routing failed: {routing_result.error_message}")
                    print("   Trying standard routing...")
                    routing_result = self.virtual_audio.route_audio(audio_result.audio_data)
            else:
                print("   VB-Cable not available, using standard routing")
                routing_result = self.virtual_audio.route_audio(audio_result.audio_data)
            
            if routing_result.success:
                print(f"   Audio routing successful!")
                print(f"      Device: {getattr(routing_result, 'device_used', 'default')}")
            else:
                print(f"   Audio routing failed: {getattr(routing_result, 'error_message', 'Unknown error')}")
                return False
            
            print(f"Processing complete! Text successfully converted to virtual microphone.")
            return True
            
        except Exception as e:
            print(f"Processing failed: {e}")
            logger.error(f"CLI text processing error: {e}")
            return False
    
    def process_file(self, file_path: str, speaker_id: int = 0, temperature: float = 1.0) -> bool:
        """
        Process text from a file through the TTS pipeline.
        
        Args:
            file_path: Path to text file
            speaker_id: Speaker ID for voice selection
            temperature: Voice temperature setting
            
        Returns:
            bool: True if processing successful
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return False
            
            if not file_path.is_file():
                print(f"❌ Path is not a file: {file_path}")
                return False
            
            print(f"📂 Reading file: {file_path}")
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin1') as f:
                    text = f.read().strip()
            
            if not text:
                print(f"❌ File is empty or contains no readable text")
                return False
            
            print(f"✅ File loaded: {len(text)} characters")
            
            # Process the text
            return self.process_text(text, speaker_id, temperature)
            
        except Exception as e:
            print(f"❌ File processing failed: {e}")
            logger.error(f"CLI file processing error: {e}")
            return False
    
    def start_interactive(self):
        """Start interactive mode for continuous text processing."""
        print("🎤 TTS Virtual Microphone - Interactive Mode")
        print("=" * 60)
        print("Enter text to convert to speech. Type 'help' for commands, 'quit' to exit.")
        print()
        
        # Default settings
        speaker_id = 0
        temperature = 1.0
        
        while True:
            try:
                # Get user input
                user_input = input("TTS> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_interactive_help()
                    continue
                
                elif user_input.lower() == 'status':
                    self._show_system_status()
                    continue
                
                elif user_input.lower().startswith('speaker '):
                    try:
                        speaker_id = int(user_input.split()[1])
                        print(f"✅ Speaker ID set to: {speaker_id}")
                    except (IndexError, ValueError):
                        print("❌ Invalid speaker ID. Usage: speaker <id>")
                    continue
                
                elif user_input.lower().startswith('temperature '):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"✅ Temperature set to: {temperature}")
                    except (IndexError, ValueError):
                        print("❌ Invalid temperature. Usage: temperature <0.1-2.0>")
                    continue
                
                # Process as text
                print()  # Add spacing
                success = self.process_text(user_input, speaker_id, temperature)
                print()  # Add spacing
                
                if not success:
                    print("⚠️  Processing failed. Try again or type 'help' for assistance.")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Interactive mode error: {e}")
    
    def _show_interactive_help(self):
        """Show help for interactive mode."""
        help_text = """
📖 Interactive Mode Commands:

TEXT PROCESSING:
    <text>              Convert text to speech
    
SETTINGS:
    speaker <id>        Set speaker ID (0, 1, 2, ...)
    temperature <val>   Set voice temperature (0.1 - 2.0)
    
SYSTEM:
    status              Show system status
    help                Show this help
    quit, exit, q       Exit interactive mode

EXAMPLES:
    Hello, world!
    speaker 1
    temperature 0.8
    This is a test with new settings
        """
        print(help_text)
    
    def _show_system_status(self):
        """Show current system status."""
        print("📊 System Status:")
        print("-" * 40)
        
        try:
            # TTS Engine status
            if hasattr(self.tts_engine, 'get_status'):
                tts_status = self.tts_engine.get_status()
                print(f"🎤 TTS Engine: {tts_status.get('status', 'Unknown')}")
                if 'loaded_plugins' in tts_status:
                    print(f"   Plugins: {', '.join(tts_status['loaded_plugins'])}")
            else:
                print("🎤 TTS Engine: Available")
            
            # Virtual Audio status
            if hasattr(self.virtual_audio, 'get_status'):
                audio_status = self.virtual_audio.get_status()
                print(f"🎛️ Virtual Audio: {audio_status.get('status', 'Unknown')}")
                if 'vb_cable_integration' in audio_status:
                    vb_status = audio_status['vb_cable_integration']
                    print(f"   VB-Cable: {'Available' if vb_status.get('available') else 'Not available'}")
            else:
                print("🎛️ Virtual Audio: Available")
            
            # Audio devices
            print("🔊 Audio Devices:")
            if hasattr(self.virtual_audio, 'get_available_devices'):
                devices = self.virtual_audio.get_available_devices()
                for device in devices[:3]:  # Show first 3 devices
                    print(f"   • {device.get('name', 'Unknown')}")
                if len(devices) > 3:
                    print(f"   ... and {len(devices) - 3} more")
            else:
                print("   Device listing not available")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    def _save_audio_file(self, audio_data: bytes, text: str):
        """Save generated audio to the output directory."""
        try:
            # Get audio output directory from config
            audio_output_dir = Path(self.config.get('tts', {}).get('sesame_csm', {}).get('audio_output_dir', './audio_output/test_results'))
            audio_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache key similar to sesame_csm plugin
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            # Create filename with timestamp for uniqueness
            timestamp = int(time.time())
            filename = f"audio_{cache_key}_{timestamp}.wav"
            filepath = audio_output_dir / filename
            
            # Save audio file
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            print(f"   💾 Audio saved: {filepath}")
            logger.debug(f"Audio file saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save audio file: {e}")
            print(f"   ⚠️  Could not save audio file: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'virtual_audio'):
                self.virtual_audio.cleanup()
            if hasattr(self, 'tts_engine'):
                self.tts_engine.cleanup()
            logger.info("CLI interface cleaned up")
        except Exception as e:
            logger.error(f"CLI cleanup error: {e}")
