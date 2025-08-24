# 🔧 TTS Virtual Microphone - Installation Guide

This guide will help you set up the TTS Virtual Microphone application from scratch.

## 📋 System Requirements

- **Operating System**: Windows 10/11 (primary support), Linux (experimental)
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 500MB for application + 2GB for models (if using advanced TTS)
- **Audio**: Sound card with speaker/headphone output

## 🚀 Quick Installation (5 minutes)

### Step 1: Check Python Installation
```bash
# Check if Python is installed
python --version

# Should show Python 3.8+ (e.g., Python 3.9.7)
```

**If Python is not installed:**
1. Download from https://python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Restart your command prompt

### Step 2: Download the Application
```bash
# Navigate to your projects folder
cd C:\Projects

# If you have the TTS folder already:
cd TTS
```

### Step 3: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Install additional API dependencies
pip install python-multipart uvicorn fastapi

# Install audio processing dependencies
pip install sounddevice soundfile numpy pydub
```

### Step 4: Test Installation
```bash
# Test basic functionality
python main.py --help

# Test text-to-speech
python main.py --text "Installation test successful!"
```

✅ **If you see help text and hear audio, you're ready to go!**

## 🎛️ VB-Cable Installation (Recommended)

VB-Cable provides the best virtual microphone experience for Discord, Zoom, gaming, etc.

### Download & Install
1. **Download**: Visit https://vb-audio.com/Cable/
2. **Extract**: Unzip the downloaded file
3. **Install**: 
   - Right-click `VBCABLE_Setup_x64.exe` (for 64-bit Windows)
   - Select "Run as administrator"
   - Follow installation prompts
4. **Reboot**: Restart your computer (required!)

### Verify Installation
```bash
# After reboot, test VB-Cable detection
python main.py --list-devices

# Should show CABLE Input/Output in the device list
```

### Configure Applications

#### For Discord:
1. Open Discord → Settings → Voice & Video
2. Set Microphone to "CABLE Output (VB-Audio Virtual Cable)"
3. Test with: `python main.py --text "Discord test"`

#### For Zoom/Teams:
1. Open Zoom → Settings → Audio
2. Set Microphone to "CABLE Output (VB-Audio Virtual Cable)"
3. Test with: `python main.py --text "Zoom test"`

#### For OBS Studio:
1. Add Audio Input Capture source
2. Select "CABLE Output (VB-Audio Virtual Cable)"
3. Test with: `python main.py --text "OBS test"`

## 📦 Detailed Dependency Installation

### Core Dependencies
```bash
# Text processing and configuration
pip install pyyaml watchdog

# Audio processing
pip install sounddevice soundfile numpy scipy

# TTS engines (current: demo plugin)
pip install torch torchaudio  # For future Sesame CSM support

# API server
pip install fastapi uvicorn python-multipart

# CLI and utilities
pip install click colorama
```

### Optional Dependencies
```bash
# Advanced audio processing
pip install pydub librosa

# Additional TTS engines
pip install pyttsx3  # Windows SAPI support

# Development tools
pip install pytest black flake8 mypy
```

### Create Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv tts_env

# Activate (Windows)
tts_env\Scripts\activate

# Activate (Linux/Mac)
source tts_env/bin/activate

# Install dependencies in virtual environment
pip install -r requirements.txt

# When done, deactivate
deactivate
```

## 🔧 Configuration Setup

### Create Configuration Directory
```bash
# Ensure config directory exists
mkdir -p config
```

### Basic Configuration (config/config.yaml)
```yaml
application:
  name: "TTS Virtual Microphone"
  version: "1.0.0"
  log_level: "INFO"

tts:
  default_engine: "demo_tts"
  fallback_engine: "pyttsx3"
  context_length: 4
  voice_settings:
    speaker_id: 0
    temperature: 1.0

audio:
  sample_rate: 24000
  channels: 1
  format: "wav"
  quality: "high"
  normalization: true

virtual_audio:
  device_name: "CABLE Input"
  auto_detect: true
  fallback_device: "default"
  buffer_size: 1024

interfaces:
  cli:
    enabled: true
  api:
    enabled: true
    port: 8080
  gui:
    enabled: false

logging:
  file: "./logs/tts_app.log"
  max_size: "10MB"
  backup_count: 5
```

## 🧪 Testing Installation

### Basic Tests
```bash
# Test 1: Application starts
python main.py --help

# Test 2: Text processing
python main.py --text "Hello world"

# Test 3: Interactive mode
python main.py --interactive
# Type: hello
# Type: quit

# Test 4: API server
python main.py --api &
# Visit: http://localhost:8080/docs
# Stop with Ctrl+C
```

### Audio Tests
```bash
# Test audio devices
python main.py --list-devices

# Test TTS engine
python main.py --test-tts

# Test virtual audio
python main.py --test-audio
```

### VB-Cable Tests
```bash
# Test VB-Cable integration
python main.py --text "VB-Cable test" --debug

# Should show in logs:
# "VB-Cable detected: X products, Y devices"
```

## 🚨 Troubleshooting Installation

### Common Installation Issues

#### Python Not Found
```
Error: 'python' is not recognized as an internal or external command
```
**Solution:**
1. Reinstall Python from python.org
2. Check "Add Python to PATH"
3. Restart command prompt
4. Try `python3` instead of `python`

#### Permission Errors
```
Error: Access is denied or permission error
```
**Solution:**
1. Run Command Prompt as Administrator
2. Or use: `python -m pip install --user -r requirements.txt`

#### Module Import Errors
```
Error: ModuleNotFoundError: No module named 'sounddevice'
```
**Solution:**
```bash
# Install missing module
pip install sounddevice

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

#### VB-Cable Not Detected
```
Warning: No suitable VB-Cable device found
```
**Solution:**
1. Ensure VB-Cable is installed
2. Reboot computer after installation
3. Check Windows Sound settings
4. Run application as Administrator

#### Audio Device Errors
```
Error: Error opening OutputStream
```
**Solution:**
1. Check audio drivers are up to date
2. Try different audio device: `python main.py --list-devices`
3. Close other audio applications
4. Restart audio service: `net stop audiosrv && net start audiosrv`

### Dependency Version Issues

#### Create requirements.txt with specific versions
```
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
sounddevice==0.4.6
soundfile==0.12.1
numpy==1.24.3
pyyaml==6.0.1
watchdog==3.0.0
pydantic==2.5.0
```

#### Install specific versions
```bash
pip install -r requirements.txt --no-deps
```

## 📁 Directory Structure Verification

After installation, your directory should look like:
```
TTS/
├── config/
│   ├── config.yaml
│   └── config_manager.py
├── core/
│   ├── audio_processor.py
│   ├── text_processor.py
│   ├── tts_engine.py
│   └── virtual_audio.py
├── interfaces/
│   ├── cli_interface.py
│   ├── api_interface.py
│   └── gui_interface.py
├── plugins/
│   └── tts/
│       └── demo_tts_plugin.py
├── utils/
│   ├── logger.py
│   └── error_handler.py
├── docs/
│   ├── USER_GUIDE.md
│   └── INSTALLATION_GUIDE.md
├── main.py
└── requirements.txt
```

## 🔄 Updating the Application

### Update Dependencies
```bash
# Update all packages
pip install -r requirements.txt --upgrade

# Update specific package
pip install fastapi --upgrade
```

### Backup Configuration
```bash
# Backup your config before updating
copy config\config.yaml config\config_backup.yaml
```

### Verify Update
```bash
# Test after update
python main.py --test-tts
python main.py --text "Update test successful"
```

## 🎯 Next Steps

After successful installation:

1. **Read the User Guide**: `docs/USER_GUIDE.md`
2. **Try Interactive Mode**: `python main.py --interactive`
3. **Test with Your Apps**: Discord, Zoom, OBS, etc.
4. **Explore API**: `python main.py --api` → http://localhost:8080/docs
5. **Customize Config**: Edit `config/config.yaml`

## 📞 Installation Support

If you encounter issues:

1. **Check Logs**: Look in `logs/tts_app.log`
2. **Run with Debug**: `python main.py --debug --text "test"`
3. **Test Components**: Use `--test-tts` and `--test-audio`
4. **Check Dependencies**: `pip list | grep -E "(fastapi|sounddevice|numpy)"`

---

## 📋 Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] VB-Cable installed and computer rebooted
- [ ] Application starts: `python main.py --help`
- [ ] Text-to-speech works: `python main.py --text "test"`
- [ ] VB-Cable detected: `python main.py --list-devices`
- [ ] Discord/Zoom configured to use CABLE Output
- [ ] API server starts: `python main.py --api`

✅ **All checked? You're ready to use TTS Virtual Microphone!**

---

*Installation Guide - Last updated: January 2025*
