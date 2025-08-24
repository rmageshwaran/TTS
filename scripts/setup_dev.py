#!/usr/bin/env python3
"""
Development Environment Setup Script

This script sets up the complete development environment for TTS Virtual Microphone.
It handles dependency installation, directory creation, and configuration.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Optional


class DevSetup:
    """Development environment setup manager."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.python_executable = sys.executable
        self.platform = platform.system().lower()
        
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        print(f"Running: {' '.join(command)}")
        try:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {e}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            if check:
                raise
            return e
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            print(f"Error: Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def check_git(self) -> bool:
        """Check if Git is available."""
        try:
            result = self.run_command(["git", "--version"], check=False)
            if result.returncode == 0:
                print("✓ Git is available")
                return True
        except FileNotFoundError:
            pass
        
        print("⚠ Git not found. Some features may not work.")
        return False
    
    def create_directories(self):
        """Create necessary project directories."""
        directories = [
            "logs",
            "models", 
            "cache",
            "temp",
            "tests/logs",
            "tests/temp",
            "tests/mock_models",
            "config/backups"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            gitkeep = dir_path / ".gitkeep"
            if not gitkeep.exists() and not any(dir_path.iterdir()):
                gitkeep.touch()
        
        print("✓ Created project directories")
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print("Installing Python dependencies...")
        
        # Upgrade pip first
        self.run_command([self.python_executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install development dependencies
        requirements_dev = self.project_root / "requirements-dev.txt"
        if requirements_dev.exists():
            self.run_command([
                self.python_executable, "-m", "pip", "install", 
                "-r", str(requirements_dev)
            ])
        
        # Install in development mode
        self.run_command([
            self.python_executable, "-m", "pip", "install", "-e", "."
        ], check=False)  # May fail if setup.py is not ready
        
        print("✓ Installed Python dependencies")
    
    def setup_pre_commit(self):
        """Setup pre-commit hooks."""
        print("Setting up pre-commit hooks...")
        
        try:
            # Install pre-commit hooks
            self.run_command(["pre-commit", "install"])
            self.run_command(["pre-commit", "install", "--hook-type", "pre-push"])
            print("✓ Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ Failed to setup pre-commit hooks. Install pre-commit manually.")
    
    def setup_git_repo(self):
        """Initialize Git repository if needed."""
        git_dir = self.project_root / ".git"
        if not git_dir.exists():
            print("Initializing Git repository...")
            try:
                os.chdir(self.project_root)
                self.run_command(["git", "init"])
                self.run_command(["git", "add", "."])
                self.run_command([
                    "git", "commit", "-m", "Initial commit: TTS Virtual Microphone project setup"
                ])
                print("✓ Git repository initialized")
            except subprocess.CalledProcessError:
                print("⚠ Failed to initialize Git repository")
        else:
            print("✓ Git repository already exists")
    
    def install_system_dependencies(self):
        """Install system-level dependencies."""
        print("Checking system dependencies...")
        
        if self.platform == "linux":
            self._install_linux_deps()
        elif self.platform == "windows":
            self._install_windows_deps()
        elif self.platform == "darwin":
            self._install_macos_deps()
        else:
            print(f"⚠ Unsupported platform: {self.platform}")
    
    def _install_linux_deps(self):
        """Install Linux system dependencies."""
        print("Linux detected. You may need to install:")
        print("  sudo apt-get install portaudio19-dev pulseaudio")
        print("  or equivalent for your distribution")
    
    def _install_windows_deps(self):
        """Install Windows system dependencies."""
        print("Windows detected.")
        print("Consider installing VB-Cable from: https://vb-audio.com/Cable/")
    
    def _install_macos_deps(self):
        """Install macOS system dependencies."""
        print("macOS detected. You may need to install:")
        print("  brew install portaudio")
    
    def verify_installation(self):
        """Verify the development environment is working."""
        print("Verifying installation...")
        
        # Test imports
        try:
            import yaml
            import numpy
            import sounddevice
            print("✓ Core dependencies can be imported")
        except ImportError as e:
            print(f"⚠ Import error: {e}")
        
        # Test basic functionality
        try:
            from config.config_manager import ConfigManager
            config_manager = ConfigManager()
            print("✓ Configuration manager can be imported")
        except Exception as e:
            print(f"⚠ Configuration manager error: {e}")
        
        # Test audio system
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print(f"✓ Found {len(devices)} audio devices")
        except Exception as e:
            print(f"⚠ Audio system error: {e}")
    
    def create_sample_config(self):
        """Create a sample development configuration."""
        config_file = self.project_root / "config" / "config-dev.yaml"
        if not config_file.exists():
            sample_config = """# Development Configuration for TTS Virtual Microphone

application:
  name: "TTS Virtual Microphone (Dev)"
  version: "1.0.0-dev"
  log_level: "DEBUG"
  debug: true

tts:
  default_engine: "sesame_csm"
  fallback_engine: "pyttsx3"
  model_path: "./models/sesame-csm-1b"

audio:
  sample_rate: 24000
  channels: 1
  format: "wav"
  quality: "high"

virtual_audio:
  device_name: "CABLE Input"
  auto_detect: true

logging:
  level: "DEBUG"
  file: "./logs/dev.log"
  console_output: true

development:
  debug_mode: true
  test_mode: false
  mock_audio_devices: false
"""
            config_file.write_text(sample_config)
            print("✓ Created development configuration")
    
    def run_setup(self):
        """Run the complete setup process."""
        print("=" * 60)
        print("TTS Virtual Microphone - Development Environment Setup")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_python_version():
            sys.exit(1)
        
        self.check_git()
        
        # Setup environment
        os.chdir(self.project_root)
        
        try:
            self.create_directories()
            self.install_system_dependencies()
            self.install_dependencies()
            self.create_sample_config()
            
            if self.check_git():
                self.setup_git_repo()
                self.setup_pre_commit()
            
            self.verify_installation()
            
            print("\n" + "=" * 60)
            print("✓ Development environment setup complete!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Review the configuration in config/config-dev.yaml")
            print("2. Install VB-Cable from https://vb-audio.com/Cable/")
            print("3. Run tests: make test-unit")
            print("4. Start development: make run")
            print("\nFor help: make help")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            print("\nPlease check the error messages above and try again.")
            sys.exit(1)


def main():
    """Main entry point."""
    setup = DevSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()