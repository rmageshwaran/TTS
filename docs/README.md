# TTS Virtual Microphone Project

## Project Overview
This project implements a Text-to-Speech (TTS) system that converts text to natural-sounding speech using Sesame CSM (Conversational Speech Model) and feeds the generated audio to the system microphone via virtual audio cable technology. The application runs completely offline on a standalone machine without requiring internet access.

## 📚 Documentation

### For Users
- **[🚀 Installation Guide](INSTALLATION_GUIDE.md)** - Complete setup instructions
- **[📖 User Guide](USER_GUIDE.md)** - Comprehensive usage guide for all interfaces
- **[⚡ Quick Reference](QUICK_REFERENCE.md)** - Essential commands and shortcuts

### For Developers
- **[🏗️ Architecture Design](architecture_design.md)** - Technical architecture and design decisions
- **[📋 Task Breakdown](project_task_breakdown.md)** - Detailed project tasks and milestones
- **[🧪 Testing Strategy](testing_strategy.md)** - Testing approach and procedures

## 🚀 Quick Start

### Installation (5 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install python-multipart uvicorn fastapi

# 2. Test installation
python main.py --help

# 3. Convert text to speech
python main.py --text "Hello, world!"
```

### For Gaming/Streaming
```bash
# Start interactive mode
python main.py --interactive

# In the TTS prompt, type your messages:
TTS> Welcome to my stream!
TTS> Thanks for the follow!
TTS> quit
```

### For Development
```bash
# Start REST API server
python main.py --api

# Visit http://localhost:8080/docs for interactive API documentation
```

**Need VB-Cable for Discord/Zoom?** See the [Installation Guide](INSTALLATION_GUIDE.md#vb-cable-installation-recommended)

## Key Features
- **Offline Operation**: No internet connectivity required
- **Natural Speech**: Utilizes Sesame CSM for highly realistic voice generation
- **Virtual Microphone**: Routes TTS output to system microphone input
- **Modular Design**: Plugin-based architecture for easy extensibility
- **Multiple Interfaces**: CLI, API, and optional GUI support
- **Fallback Support**: Multiple TTS engines and audio routing options

## Architecture Highlights

### Modular Plugin System
- **TTS Plugins**: Sesame CSM (primary), fallback engines (pyttsx3, SAPI)
- **Audio Plugins**: Effects processing, format conversion
- **Interface Plugins**: CLI, REST API, GUI options

### Virtual Audio Integration
- **Primary**: VB-Cable virtual audio driver
- **Fallbacks**: Windows Sound Mixer, file output
- **Auto-detection**: Automatic device discovery and configuration

### Real-time Processing
- **Low Latency**: <500ms end-to-end processing time
- **Streaming**: Real-time audio streaming to virtual microphone
- **Context Aware**: Conversational context support for natural speech

## Technology Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **Sesame CSM**: Primary TTS model (Apache 2.0 license)
- **PyTorch**: Deep learning framework for model inference
- **SoundDevice**: Audio I/O handling
- **VB-Cable**: Virtual audio cable driver

### Dependencies
```
torch>=2.0.0
transformers>=4.52.1
sounddevice>=0.4.6
numpy>=1.21.0
pyyaml>=6.0
requests>=2.28.0
fastapi>=0.68.0  # For API interface
uvicorn>=0.15.0  # For API server
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (primary), Linux (experimental)
- **Python**: 3.10 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models and application
- **Audio**: Sound card with audio input/output

### Recommended Requirements
- **GPU**: CUDA-compatible GPU for optimal performance
- **RAM**: 16GB for best performance
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **Storage**: SSD for faster model loading

## Installation Guide

### 1. System Prerequisites
```bash
# Install Python 3.10+
# Download from https://python.org

# Install VB-Cable
# Download from https://vb-audio.com/Cable/
```

### 2. Application Setup
```bash
# Clone the repository
git clone <repository-url>
cd tts_virtual_mic

# Install Python dependencies
pip install -r requirements.txt

# Download Sesame CSM model
python scripts/download_models.py

# Setup virtual audio
python scripts/setup_virtual_audio.py
```

### 3. Configuration
```bash
# Copy and edit configuration
cp config/config.yaml.example config/config.yaml

# Test the setup
python main.py --test
```

## Usage Examples

### Command Line Interface
```bash
# Basic text-to-speech
python main.py --text "Hello, this is a test message"

# Interactive mode
python main.py --interactive

# From file
python main.py --file input.txt

# With custom voice settings
python main.py --text "Hello world" --speaker 0 --temperature 1.2
```

### REST API
```bash
# Start API server
python main.py --api --port 8080

# Send text via curl
curl -X POST http://localhost:8080/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "speaker_id": 0}'
```

### Python API
```python
from tts_virtual_mic import TTSVirtualMic

# Initialize the TTS system
tts = TTSVirtualMic()

# Generate speech
tts.speak("Hello, this is generated speech")

# With context for natural conversation
tts.speak("How are you today?", context=previous_conversation)
```

## Configuration Options

### TTS Settings
```yaml
tts:
  default_engine: "sesame_csm"
  model_path: "./models/sesame-csm-1b"
  voice_settings:
    speaker_id: 0
    temperature: 1.0
    max_audio_length_ms: 10000
```

### Audio Settings
```yaml
audio:
  sample_rate: 24000
  channels: 1
  format: "wav"
  quality: "high"
  normalization: true
```

### Virtual Audio Settings
```yaml
virtual_audio:
  device_name: "CABLE Input"
  auto_detect: true
  buffer_size: 1024
```

## Sesame CSM Information

### Model Capabilities
- **Architecture**: Llama backbone + audio decoder
- **Quality**: Highly natural, human-like speech
- **Features**: Context awareness, emotional expression
- **Languages**: Primarily English (with limited multilingual support)
- **Latency**: <500ms end-to-end generation time

### Model Advantages
- **Naturalness**: Superior to traditional TTS systems
- **Context**: Conversational awareness for natural flow
- **Emotions**: Capable of emotional expression
- **Offline**: Complete offline operation
- **Open Source**: Apache 2.0 license

### Model Limitations
- **Size**: 1B parameters (manageable but not tiny)
- **GPU**: Benefits significantly from CUDA acceleration
- **Languages**: Limited non-English language support
- **Training**: Base model requires fine-tuning for specific voices

## Virtual Audio Cable Setup

### VB-Cable Installation
1. Download VB-Cable from official website
2. Run installer as administrator
3. Restart system after installation
4. Verify installation in Windows Sound settings

### Configuration Steps
1. Set application audio output to "CABLE Input"
2. Set target application microphone to "CABLE Output"
3. Test audio routing with the included test script

### Troubleshooting
- Ensure VB-Cable devices are not set as default system audio
- Check for device conflicts in Windows Sound settings
- Verify audio permissions for the application
- Test with simple audio file playback first

## Application Architecture

### Core Components
1. **Text Processor**: Input validation and preprocessing
2. **TTS Engine Manager**: Model loading and inference coordination
3. **Audio Processor**: Format conversion and quality control
4. **Virtual Audio Interface**: Routing to system microphone

### Plugin System
- **TTS Plugins**: Modular TTS engine support
- **Audio Plugins**: Effects and processing plugins
- **Interface Plugins**: Multiple user interface options

### Error Handling
- **Graceful Degradation**: Fallback to simpler TTS engines
- **Device Recovery**: Automatic reconnection to audio devices
- **Logging**: Comprehensive error logging and debugging

## Performance Optimization

### Model Optimization
- **Quantization**: Support for model quantization
- **Caching**: Model caching for faster startup
- **Batching**: Batch processing for multiple requests

### Audio Optimization
- **Streaming**: Real-time audio streaming
- **Buffering**: Optimized buffer management
- **Threading**: Multi-threaded audio processing

### Memory Management
- **Lazy Loading**: Load models only when needed
- **Cleanup**: Automatic cleanup of temporary files
- **Monitoring**: Memory usage monitoring and alerts

## Security and Privacy

### Data Protection
- **Local Processing**: All processing happens locally
- **No Network**: No data transmission over network
- **Temporary Files**: Automatic cleanup of temporary audio files
- **Memory**: Secure memory handling for sensitive text

### Model Security
- **Verification**: Model integrity verification
- **Sandboxing**: Isolated model execution environment
- **Updates**: Secure model update mechanism

## Testing and Quality Assurance

### Automated Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Latency and quality benchmarks

### Manual Testing
- **Audio Quality**: Subjective audio quality evaluation
- **Device Compatibility**: Testing with various audio devices
- **OS Compatibility**: Cross-platform compatibility testing

## Troubleshooting Guide

### Common Issues
1. **VB-Cable not detected**: Verify installation and restart
2. **Poor audio quality**: Check sample rate and format settings
3. **High latency**: Optimize buffer sizes and GPU usage
4. **Model loading errors**: Verify model files and permissions

### Diagnostic Tools
```bash
# Test audio devices
python main.py --list-devices

# Test TTS engine
python main.py --test-tts

# Test virtual audio
python main.py --test-audio

# Verbose logging
python main.py --debug --text "test"
```

## Future Enhancements

### Planned Features
- **Voice Cloning**: Custom voice training support
- **Real-time Voice Changing**: Live voice modification
- **Multiple Languages**: Extended language support
- **Cloud Sync**: Optional cloud synchronization for settings

### Plugin Ecosystem
- **Community Plugins**: Support for community-developed plugins
- **Voice Effects**: Audio effects and filters
- **Integration Plugins**: Direct integration with popular applications

## Contributing

### Development Setup
```bash
# Clone development branch
git clone -b develop <repository-url>

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

### Plugin Development
- Follow the plugin interface specifications
- Include comprehensive tests
- Document configuration options
- Provide usage examples

## License and Legal

### Software License
- **Application**: MIT License
- **Sesame CSM**: Apache 2.0 License
- **Dependencies**: Various open-source licenses

### Usage Guidelines
- Respect voice cloning ethical guidelines
- Obtain consent for voice replication
- Follow local laws regarding synthetic speech
- Avoid malicious use cases

## Support and Community

### Documentation
- **Full Documentation**: Available in `docs/` directory
- **API Reference**: Generated from code documentation
- **Examples**: Comprehensive examples in `examples/` directory

### Community Support
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussions**: Community discussions and Q&A
- **Wiki**: Community-maintained documentation and tips

### Professional Support
- **Commercial License**: Available for commercial deployments
- **Custom Development**: Professional services for custom features
- **Training**: Training and consultation services

---

This architecture provides a solid foundation for building a robust, offline TTS virtual microphone system using cutting-edge Sesame CSM technology while maintaining modularity and extensibility for future enhancements.