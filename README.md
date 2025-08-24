# TTS Virtual Microphone

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Offline Operation](https://img.shields.io/badge/Operation-Offline-green.svg)](https://github.com/your-org/tts-virtual-mic)

**Enterprise-grade Text-to-Speech system with virtual microphone routing for completely offline, intranet-based environments.**

Convert text to high-quality speech and route it seamlessly to virtual microphone devices for video conferencing, streaming, and communication applications.

## 🎯 Key Features

### 🔒 **Offline & Secure**
- **Complete offline operation** - No internet connectivity required
- **Intranet-optimized** - Designed for secure, isolated networks
- **Local model storage** - All AI models cached locally
- **Input validation & sanitization** - Enterprise security standards

### 🎤 **Advanced TTS Capabilities**
- **Sesame CSM Integration** - High-quality conversational speech model
- **Voice cloning support** - Custom voice training and embeddings
- **Context-aware generation** - Maintains conversation context
- **Multi-language support** - Automatic language detection
- **SSML processing** - Advanced speech markup support

### 🎛️ **Virtual Audio Excellence**
- **VB-Cable integration** - Direct virtual microphone routing
- **Auto-device detection** - Intelligent audio device management
- **Low-latency streaming** - Optimized for real-time applications
- **Multiple fallback options** - Robust error recovery
- **Performance monitoring** - Real-time metrics and health checks

### 🔧 **Enterprise Architecture**
- **Plugin system** - Extensible TTS engine support
- **Load balancing** - Intelligent engine selection
- **Health monitoring** - Automatic failure detection and recovery
- **Comprehensive logging** - Detailed system diagnostics
- **Multiple interfaces** - CLI, REST API, and GUI support

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (Required)
- **VB-Audio Virtual Cable** (Recommended for best experience)
  - Download from: https://vb-audio.com/Cable/
- **Windows/Linux/macOS** (Cross-platform support)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/tts-virtual-mic.git
   cd tts-virtual-mic
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download TTS models** (automatic on first run):
   ```bash
   python scripts/download_models.py
   ```

### Basic Usage

#### Command Line Interface

```bash
# Convert text to speech
python main.py --text "Hello, world!"

# Process text file
python main.py --file input.txt

# Interactive mode
python main.py --interactive

# Use specific voice settings
python main.py --text "Hello" --speaker 1 --temperature 0.8
```

#### REST API Server

```bash
# Start API server
python main.py --api --port 8080

# Access interactive documentation
# Visit: http://localhost:8080/docs
```

#### API Usage Example

```python
import requests

# Convert text to speech via API
response = requests.post("http://localhost:8080/tts", json={
    "text": "Hello from the TTS API!",
    "speaker_id": 0,
    "temperature": 1.0
})

print(response.json())
```

## 📋 System Requirements

### Minimum Requirements
- **CPU:** Dual-core 2.0 GHz
- **RAM:** 4 GB
- **Storage:** 2 GB free space
- **OS:** Windows 10/Linux/macOS 10.14+

### Recommended Requirements
- **CPU:** Quad-core 3.0 GHz or higher
- **RAM:** 8 GB or more
- **GPU:** NVIDIA GPU with CUDA support (optional, for acceleration)
- **Storage:** 5 GB free space (for model caching)
- **Audio:** VB-Audio Virtual Cable installed

## 🏗️ Architecture Overview

The TTS Virtual Microphone follows a modular, layered architecture designed for enterprise reliability and extensibility:

### Core Components

1. **Text Processing Layer**
   - Advanced text normalization and validation
   - SSML support for speech control
   - Multi-language detection and processing

2. **TTS Engine Layer**
   - Plugin-based architecture supporting multiple engines
   - Load balancing and health monitoring
   - Context management for conversational TTS

3. **Audio Processing Layer**
   - Format conversion and quality control
   - Real-time effects and normalization
   - Streaming and buffer management

4. **Virtual Audio Layer**
   - VB-Cable integration with auto-detection
   - Multiple routing strategies and optimization
   - Performance monitoring and metrics

5. **Interface Layer**
   - CLI for command-line operation
   - REST API for programmatic access
   - GUI for user-friendly interaction (planned)

## ⚙️ Configuration

The system uses a comprehensive YAML configuration file located at `config/config.yaml`. Key configuration sections:

### TTS Engine Configuration
```yaml
tts:
  default_engine: "sesame_csm"
  fallback_engine: "demo_tts"
  model_path: "./models/sesame-csm-1b"
  context_length: 4
  load_balancing: true
```

### Audio Settings
```yaml
audio:
  sample_rate: 24000
  channels: 1
  format: "wav"
  normalization: true
  effects:
    enabled: false
    reverb: 0.1
```

### Virtual Audio Configuration
```yaml
virtual_audio:
  device_name: "CABLE Input"
  auto_detect: true
  latency_mode: "low"
  enable_monitoring: true
```

### API Configuration
```yaml
interfaces:
  api:
    enabled: true
    host: "localhost"
    port: 8080
    cors_enabled: true
    rate_limiting: true
```

## 🔌 Plugin System

The TTS Virtual Microphone supports a flexible plugin architecture for extending functionality:

### Available TTS Plugins

1. **Sesame CSM Plugin** (`sesame_csm`)
   - High-quality conversational speech model
   - Voice cloning and custom voices
   - Context-aware generation
   - GPU acceleration support

2. **Demo TTS Plugin** (`demo_tts`)
   - Lightweight synthetic speech
   - No external dependencies
   - Testing and fallback purposes

### Creating Custom Plugins

```python
from core.tts_engine import BaseTTSPlugin

class CustomTTSPlugin(BaseTTSPlugin):
    def initialize(self, config):
        # Plugin initialization
        return True
    
    def generate_speech(self, text, speaker_id=None, context=None):
        # Generate audio bytes
        return audio_bytes
    
    def is_available(self):
        # Check plugin availability
        return True
```

## 🎛️ Virtual Audio Setup

### VB-Audio Virtual Cable Setup

1. **Download and install VB-Audio Virtual Cable:**
   - Visit: https://vb-audio.com/Cable/
   - Download the appropriate version for your OS
   - Install following the provided instructions

2. **Verify installation:**
   ```bash
   python main.py --list-devices
   ```

3. **Configure applications:**
   - Set "CABLE Input" as microphone in your target application
   - The TTS system will route audio to this virtual device

### Alternative Virtual Audio Solutions

- **VoiceMeeter** (Windows) - Advanced virtual audio mixer
- **PulseAudio** (Linux) - Built-in virtual audio support
- **Soundflower** (macOS) - Virtual audio routing

## 📊 Performance & Monitoring

### Performance Metrics

The system provides comprehensive performance monitoring:

- **TTS Engine Metrics:** Generation time, success rate, cache hit ratio
- **Audio Routing Metrics:** Latency, buffer health, device status
- **System Metrics:** Memory usage, CPU utilization, error rates

### Health Monitoring

```bash
# Check system status
curl http://localhost:8080/status

# Get detailed metrics
curl http://localhost:8080/health
```

### Optimization Tips

1. **GPU Acceleration:** Install CUDA for faster TTS generation
2. **Model Caching:** Pre-download models to reduce first-run latency
3. **Buffer Tuning:** Adjust buffer sizes for your latency requirements
4. **Device Selection:** Use dedicated virtual audio devices for best performance

## 🛠️ Development

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 core/ interfaces/ plugins/
black core/ interfaces/ plugins/
```

### Project Structure

```
TTS/
├── config/                 # Configuration management
├── core/                   # Core processing modules
│   ├── audio_processor.py  # Audio processing and effects
│   ├── text_processor.py   # Text normalization and validation
│   ├── tts_engine.py      # TTS engine management
│   ├── virtual_audio.py   # Virtual audio interface
│   └── vb_cable_interface.py # VB-Cable integration
├── interfaces/             # User interfaces
│   ├── cli_interface.py   # Command-line interface
│   ├── api_interface.py   # REST API interface
│   └── gui_interface.py   # GUI interface (planned)
├── plugins/               # Plugin system
│   ├── tts/              # TTS engine plugins
│   └── audio/            # Audio processing plugins
├── utils/                # Utility modules
├── tests/                # Test suite
└── scripts/              # Utility scripts
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# Run with coverage
pytest --cov=core --cov=interfaces --cov=plugins
```

## 🔧 Troubleshooting

### Common Issues

#### VB-Cable Not Detected
```bash
# Check available devices
python main.py --list-devices

# Test VB-Cable installation
python main.py --test-audio
```

#### TTS Model Loading Issues
```bash
# Re-download models
python scripts/download_models.py --force

# Check model status
python main.py --test-tts
```

#### Audio Routing Problems
```bash
# Test audio devices
python main.py --test-audio

# Check system status
python main.py --api &
curl http://localhost:8080/status
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python main.py --debug --verbose --text "Test message"
```

### Log Files

- **Application logs:** `./logs/tts_app.log`
- **Error logs:** `./logs/errors.log`
- **Performance logs:** `./logs/performance.log`

## 📚 API Reference

### REST API Endpoints

#### Convert Text to Speech
```http
POST /tts
Content-Type: application/json

{
    "text": "Hello, world!",
    "speaker_id": 0,
    "temperature": 1.0,
    "output_format": "wav"
}
```

#### Upload File for TTS
```http
POST /tts/file
Content-Type: multipart/form-data

file: [text file]
speaker_id: 0
temperature: 1.0
```

#### System Status
```http
GET /status
```

#### Health Check
```http
GET /health
```

#### List Audio Devices
```http
GET /devices
```

### CLI Commands

```bash
# Text processing
python main.py --text "Your text here"
python main.py --file input.txt
python main.py --interactive

# Interface options
python main.py --api --port 8080
python main.py --gui

# Voice settings
python main.py --text "Hello" --speaker 1 --temperature 0.8

# System utilities
python main.py --test-tts
python main.py --test-audio
python main.py --list-devices

# Configuration
python main.py --config custom_config.yaml
python main.py --debug --verbose
```

## 🤝 Contributing

We welcome contributions to the TTS Virtual Microphone project! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sesame Team** - For the excellent CSM conversational speech model
- **VB-Audio** - For the Virtual Cable technology
- **HuggingFace** - For the transformers library and model hosting
- **Contributors** - All the developers who have contributed to this project

## 📞 Support

- **Documentation:** [Project Wiki](https://github.com/your-org/tts-virtual-mic/wiki)
- **Issues:** [GitHub Issues](https://github.com/your-org/tts-virtual-mic/issues)
- **Discussions:** [GitHub Discussions](https://github.com/your-org/tts-virtual-mic/discussions)

---

**Built with ❤️ for offline, enterprise environments**
