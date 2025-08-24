# 🚀 TTS Virtual Microphone - Quick Reference

## 📋 Essential Commands

### Basic Usage
```bash
# Convert text to speech
python main.py --text "Your message here"

# Interactive mode (recommended for continuous use)
python main.py --interactive

# Process text file
python main.py --file script.txt

# Get help
python main.py --help
```

### Interface Options
```bash
# Command Line Interface
python main.py --text "Hello"

# REST API Server
python main.py --api --port 8080

# GUI Interface (stub)
python main.py --gui
```

## 🎮 Interactive Mode Commands

```
TTS> Your text here                     # Convert to speech
TTS> speaker 1                          # Change speaker ID
TTS> temperature 0.8                    # Adjust voice temperature
TTS> status                             # System status
TTS> help                               # Show help
TTS> quit                               # Exit
```

## 🌐 API Endpoints

**Base URL**: `http://localhost:8080`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/tts` | POST | Convert text to speech |
| `/tts/file` | POST | Upload and convert file |
| `/devices` | GET | List audio devices |
| `/docs` | GET | Interactive API docs |

### API Examples

#### Convert Text
```bash
curl -X POST "http://localhost:8080/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello API!", "speaker_id": 0}'
```

#### Upload File
```bash
curl -X POST "http://localhost:8080/tts/file" \
  -F "file=@script.txt" \
  -F "speaker_id=0"
```

#### Check Status
```bash
curl "http://localhost:8080/status"
```

## ⚙️ Configuration Quick Edit

**File**: `config/config.yaml`

```yaml
# Essential settings
tts:
  default_engine: "demo_tts"
  voice_settings:
    speaker_id: 0
    temperature: 1.0

audio:
  sample_rate: 24000          # Higher = better quality
  channels: 1                 # 1=mono, 2=stereo
  format: "wav"

virtual_audio:
  device_name: "CABLE Input"  # VB-Cable device
  auto_detect: true
```

## 🎯 Use Case Shortcuts

### Gaming/Streaming
```bash
# Start and keep running
python main.py --interactive

# Quick messages
TTS> Welcome to my stream!
TTS> Thanks for the follow!
TTS> Don't forget to subscribe!
```

### Discord/Voice Chat
```bash
# Set Discord mic to "CABLE Output"
python main.py --interactive

TTS> Hello everyone!
TTS> How's everyone doing today?
```

### Content Creation
```bash
# Process entire script
python main.py --file voiceover.txt

# Or segment by segment
python main.py --interactive
TTS> Welcome to today's tutorial
TTS> In this video we'll cover...
```

### Development Integration
```python
import requests

# Python integration
response = requests.post('http://localhost:8080/tts', 
                        json={'text': 'Build completed!'})
```

## 🔧 Troubleshooting Quick Fixes

### App Won't Start
```bash
pip install -r requirements.txt
python main.py --help
```

### No Audio
```bash
python main.py --list-devices
# Install VB-Cable if not listed
```

### API Errors
```bash
pip install python-multipart
python main.py --api
```

### VB-Cable Issues
1. Install from https://vb-audio.com/Cable/
2. Reboot computer
3. Run as Administrator

## 📱 Application Setup

### Discord Setup
1. Settings → Voice & Video
2. Microphone → "CABLE Output (VB-Audio Virtual Cable)"
3. Test: `python main.py --text "Discord test"`

### OBS Setup
1. Add "Audio Input Capture"
2. Device → "CABLE Output (VB-Audio Virtual Cable)"
3. Test: `python main.py --text "OBS test"`

### Zoom Setup
1. Settings → Audio
2. Microphone → "CABLE Output (VB-Audio Virtual Cable)"
3. Test: `python main.py --text "Zoom test"`

## 🎛️ Voice Settings

```bash
# Different speakers (if available)
python main.py --text "Hello" --speaker 0
python main.py --text "Hello" --speaker 1

# Voice temperature (0.1-2.0)
python main.py --text "Robotic" --temperature 0.3
python main.py --text "Natural" --temperature 1.5
```

## 🚨 Emergency Commands

```bash
# Test everything
python main.py --test

# Debug mode
python main.py --debug --text "debug test"

# List all devices
python main.py --list-devices

# System status
python main.py --interactive
TTS> status
```

## 📊 Performance Tips

- **Faster**: Use shorter text segments
- **Quality**: Increase sample_rate to 48000
- **Stability**: Restart app periodically in long sessions
- **Memory**: Close other audio apps

## 🔗 Important URLs

- **API Docs**: http://localhost:8080/docs
- **VB-Cable**: https://vb-audio.com/Cable/
- **Config File**: `config/config.yaml`
- **Logs**: `logs/tts_app.log`

## 📋 Keyboard Shortcuts

### Interactive Mode
- **Enter**: Process current text
- **Ctrl+C**: Exit interactive mode
- **Up/Down**: Command history (if supported)

### API Testing
- **Ctrl+C**: Stop API server
- **F5**: Refresh API docs page

---

## 🎯 Most Common Commands

```bash
# The Big 3 for daily use:
python main.py --interactive          # Gaming/streaming
python main.py --api                  # Development
python main.py --text "Quick message" # One-off messages
```

---

*Quick Reference - Last updated: January 2025*
