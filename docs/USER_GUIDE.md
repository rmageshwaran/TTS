# 🎤 TTS Virtual Microphone - Complete User Guide

Welcome to the TTS Virtual Microphone application! This guide will help you convert text to speech and route it to virtual microphone devices for use in Discord, Zoom, gaming, streaming, and other applications.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation & Setup](#-installation--setup)
- [Interface Options](#-interface-options)
- [Command Line Interface (CLI)](#-command-line-interface-cli)
- [REST API Interface](#-rest-api-interface)
- [Configuration](#-configuration)
- [Use Cases & Examples](#-use-cases--examples)
- [Troubleshooting](#-troubleshooting)
- [Advanced Features](#-advanced-features)

---

## 🚀 Quick Start

### For Beginners (5 minutes)
```bash
# 1. Convert simple text to speech
python main.py --text "Hello, this is my virtual voice!"

# 2. Start interactive mode for continuous use
python main.py --interactive

# 3. Process a text file
python main.py --file my_script.txt
```

### For Gamers & Streamers
```bash
# Start interactive mode and keep it running
python main.py --interactive

# Then type text that will be spoken through your virtual microphone
TTS> Welcome to my stream, everyone!
TTS> Don't forget to like and subscribe!
```

### For Developers
```bash
# Start the REST API server
python main.py --api

# Then visit http://localhost:8080/docs for interactive API documentation
```

---

## 🔧 Installation & Setup

### Prerequisites
1. **Python 3.8+** installed on your system
2. **VB-Audio Virtual Cable** (recommended for best results)
   - Download from: https://vb-audio.com/Cable/
   - Install and reboot your computer

### Basic Installation
```bash
# Navigate to the TTS application folder
cd C:\Projects\TTS

# Install dependencies (if needed)
pip install -r requirements.txt

# Test the installation
python main.py --help
```

### VB-Cable Setup (Recommended)
1. **Download VB-Cable**: Visit https://vb-audio.com/Cable/
2. **Install**: Run the installer as Administrator
3. **Reboot**: Restart your computer (required)
4. **Verify**: Check Windows Sound settings for "CABLE Input" device

---

## 🎯 Interface Options

The application offers three ways to use it:

| Interface | Best For | Command |
|-----------|----------|---------|
| **CLI** | Quick tasks, scripts, beginners | `python main.py` |
| **Interactive** | Gaming, streaming, continuous use | `python main.py --interactive` |
| **API** | Developers, integrations, web apps | `python main.py --api` |
| **GUI** | *(Coming soon)* | `python main.py --gui` |

---

## 💻 Command Line Interface (CLI)

### Basic Commands

#### Convert Text to Speech
```bash
# Simple text conversion
python main.py --text "Hello world!"

# Longer text with quotes
python main.py --text "This is a longer message that will be converted to speech and sent to your virtual microphone."

# Text with special characters
python main.py --text "Welcome to my stream! Don't forget to subscribe 😊"
```

#### Process Text Files
```bash
# Convert a text file
python main.py --file script.txt

# Process multiple paragraphs
python main.py --file announcement.txt
```

**Example text file (`script.txt`):**
```
Welcome everyone to today's presentation.
We'll be covering three main topics.
Please feel free to ask questions at the end.
```

#### Voice Settings
```bash
# Use different speaker voice (if available)
python main.py --text "Hello" --speaker 1

# Adjust voice temperature (0.1-2.0, default 1.0)
python main.py --text "Hello" --temperature 0.8
```

### Interactive Mode (Recommended for Continuous Use)

Interactive mode is perfect for gaming, streaming, or any situation where you need to convert text to speech multiple times.

```bash
# Start interactive mode
python main.py --interactive
```

**Interactive Commands:**
```
TTS> Hello everyone!                    # Convert text to speech
TTS> speaker 1                          # Change speaker voice
TTS> temperature 0.8                    # Adjust voice settings
TTS> status                             # Check system status
TTS> help                               # Show help
TTS> quit                               # Exit interactive mode
```

**Real Gaming Example:**
```
TTS> Welcome to my stream!
TTS> Today we're playing Minecraft
TTS> Don't forget to hit that subscribe button
TTS> speaker 2                          # Switch to different voice
TTS> Thanks for the follow, John!
```

### Command Line Options Reference

```bash
python main.py [OPTIONS]

Text Input:
  --text, -t TEXT         Text to convert to speech
  --file, -f FILE         File containing text to convert
  --interactive, -i       Start interactive mode

Voice Settings:
  --speaker ID            Speaker ID (0, 1, 2, ...)
  --temperature VALUE     Voice temperature (0.1-2.0)

Interface Options:
  --api                   Start REST API server
  --gui                   Start GUI interface
  --port PORT             API server port (default: 8080)

System Options:
  --config, -c FILE       Configuration file path
  --debug, -d             Enable debug logging
  --verbose, -v           Verbose output
  --help, -h              Show help message

Testing & Diagnostics:
  --test                  Run system tests
  --test-tts              Test TTS engine
  --test-audio            Test virtual audio
  --list-devices          List available audio devices
```

---

## 🌐 REST API Interface

The REST API is perfect for developers who want to integrate TTS functionality into web applications, mobile apps, or other software.

### Starting the API Server
```bash
# Start API server (default port 8080)
python main.py --api

# Start on custom port
python main.py --api --port 9000
```

### Interactive API Documentation
Once the server is running, visit:
- **API Docs**: http://localhost:8080/docs
- **Alternative Docs**: http://localhost:8080/redoc

### API Endpoints

#### Convert Text to Speech
```http
POST /tts
Content-Type: application/json

{
  "text": "Hello, this is a test",
  "speaker_id": 0,
  "temperature": 1.0,
  "output_format": "wav"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Text successfully converted to speech and routed to virtual microphone",
  "audio_length_ms": 2100,
  "audio_size_bytes": 92654,
  "processing_time_ms": 45.2,
  "device_used": "VB-Cable"
}
```

#### Upload Text File
```http
POST /tts/file
Content-Type: multipart/form-data

file: [your_text_file.txt]
speaker_id: 0
temperature: 1.0
```

#### System Status
```http
GET /status
```

**Response:**
```json
{
  "status": "healthy",
  "tts_engine": {
    "status": "available",
    "loaded_plugins": ["demo_tts"]
  },
  "virtual_audio": {
    "status": "available",
    "vb_cable_available": true
  },
  "available_devices": [...]
}
```

#### Health Check
```http
GET /health
```

### API Usage Examples

#### Python Example
```python
import requests

# Convert text to speech
response = requests.post('http://localhost:8080/tts', json={
    'text': 'Hello from Python!',
    'speaker_id': 0,
    'temperature': 1.0
})

result = response.json()
print(f"Success: {result['success']}")
print(f"Processing time: {result['processing_time_ms']}ms")
```

#### JavaScript Example
```javascript
// Convert text to speech
fetch('http://localhost:8080/tts', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Hello from JavaScript!',
    speaker_id: 0,
    temperature: 1.0
  })
})
.then(response => response.json())
.then(data => {
  console.log('Success:', data.success);
  console.log('Processing time:', data.processing_time_ms, 'ms');
});
```

#### curl Example
```bash
# Convert text to speech
curl -X POST "http://localhost:8080/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from curl!", "speaker_id": 0, "temperature": 1.0}'

# Upload file
curl -X POST "http://localhost:8080/tts/file" \
  -F "file=@script.txt" \
  -F "speaker_id=0" \
  -F "temperature=1.0"

# Check status
curl "http://localhost:8080/status"
```

---

## ⚙️ Configuration

### Configuration File Location
The application uses `config/config.yaml` for settings. You can also specify a custom config file:

```bash
python main.py --config my_custom_config.yaml --text "Hello"
```

### Key Configuration Settings

#### TTS Engine Settings
```yaml
tts:
  default_engine: "demo_tts"           # Primary TTS engine
  fallback_engine: "pyttsx3"          # Backup engine
  context_length: 4                   # Conversation context
  voice_settings:
    speaker_id: 0                      # Default speaker
    temperature: 1.0                   # Default voice temperature
```

#### Audio Settings
```yaml
audio:
  sample_rate: 24000                   # Audio quality (Hz)
  channels: 1                          # Mono (1) or Stereo (2)
  format: "wav"                        # Audio format
  quality: "high"                      # Audio quality level
  normalization: true                  # Auto-normalize volume
```

#### Virtual Audio Settings
```yaml
virtual_audio:
  device_name: "CABLE Input"           # VB-Cable device name
  auto_detect: true                    # Auto-detect VB-Cable
  fallback_device: "default"          # Fallback audio device
  buffer_size: 1024                    # Audio buffer size
```

---

## 🎮 Use Cases & Examples

### 1. Gaming & Streaming

**Scenario**: You're streaming a game and want to respond to chat with generated speech.

```bash
# Start interactive mode
python main.py --interactive

# During your stream:
TTS> Welcome to the stream, everyone!
TTS> Thanks for the follow, @username!
TTS> Don't forget to like and subscribe!
TTS> Let's check out this new area
```

**Pro Tips:**
- Keep the interactive window open beside your game
- Use short, clear phrases for best results
- Set up hotkeys in OBS to mute/unmute your virtual microphone

### 2. Discord Voice Chat

**Scenario**: Join Discord voice channels with text-to-speech responses.

**Setup:**
1. Set your Discord microphone to "CABLE Output (VB-Audio Virtual Cable)"
2. Start the TTS application in interactive mode
3. Type messages that will be heard in Discord

```bash
python main.py --interactive

TTS> Hello Discord friends!
TTS> speaker 1                    # Try different voices
TTS> This is my virtual voice
```

### 3. Zoom/Teams Meetings

**Scenario**: Participate in meetings with generated speech (useful for accessibility).

**Setup:**
1. In Zoom/Teams, select "CABLE Output" as your microphone
2. Keep your real microphone muted
3. Use the TTS application to speak

```bash
python main.py --interactive

TTS> Good morning everyone
TTS> I agree with that proposal
TTS> Could you repeat the last point?
```

### 4. Content Creation

**Scenario**: Create voiceovers for videos or presentations.

```bash
# Process an entire script
python main.py --file voiceover_script.txt

# Or process segments interactively
python main.py --interactive

TTS> Welcome to today's tutorial
TTS> In this video, we'll learn about...
TTS> First, let's look at the basics
```

### 5. Accessibility

**Scenario**: Use text-to-speech for communication assistance.

```bash
# Quick responses
python main.py --text "Yes, I agree"
python main.py --text "Could you please repeat that?"
python main.py --text "Thank you for your help"

# Interactive for conversations
python main.py --interactive
```

### 6. Automation & Scripts

**Scenario**: Integrate TTS into your automation scripts.

```python
import subprocess
import requests

# Using CLI from Python
def speak_text(text):
    subprocess.run(['python', 'main.py', '--text', text])

# Using API from Python
def speak_via_api(text):
    response = requests.post('http://localhost:8080/tts', 
                           json={'text': text})
    return response.json()

# Examples
speak_text("Build completed successfully")
speak_via_api("New email received")
```

### 7. Language Learning

**Scenario**: Practice pronunciation by listening to text being spoken.

```bash
python main.py --interactive

TTS> Hello, how are you today?
TTS> The weather is very nice
TTS> Where is the nearest restaurant?
TTS> speaker 2                    # Try different voices
TTS> Thank you very much
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Application Won't Start
```
Error: ImportError or module not found
```
**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version (need 3.8+)
python --version
```

#### 2. No Audio Output
```
Error: Audio routing failed
```
**Solutions:**
- **Install VB-Cable**: Download from https://vb-audio.com/Cable/
- **Reboot**: Restart computer after VB-Cable installation
- **Check devices**: `python main.py --list-devices`
- **Try fallback**: Application will attempt standard audio devices

#### 3. VB-Cable Not Detected
```
Warning: No suitable VB-Cable device found
```
**Solutions:**
- Install VB-Cable software
- Reboot your computer
- Check Windows Sound settings for "CABLE Input/Output"
- Try running as Administrator

#### 4. API Server Issues
```
Error: Form data requires "python-multipart"
```
**Solution:**
```bash
pip install python-multipart
```

#### 5. Permission Errors
```
Error: Access denied or permission error
```
**Solutions:**
- Run Command Prompt as Administrator
- Check file permissions in project directory
- Ensure antivirus isn't blocking the application

#### 6. Poor Audio Quality
**Solutions:**
- Increase sample rate in config: `sample_rate: 48000`
- Enable normalization: `normalization: true`
- Try different audio format: `format: "wav"`

### Debugging Commands

```bash
# Test TTS engine
python main.py --test-tts

# Test virtual audio
python main.py --test-audio

# List available audio devices
python main.py --list-devices

# Enable debug logging
python main.py --debug --text "test"

# Check system status
python main.py --interactive
TTS> status
```

### Performance Tips

1. **Faster Processing**: Use shorter text segments
2. **Better Quality**: Increase sample rate in config
3. **Lower Latency**: Reduce buffer sizes in config
4. **Memory Usage**: Restart application periodically for long sessions

---

## 🚀 Advanced Features

### Multiple Speaker Voices
```bash
# Try different speakers (if available)
python main.py --text "Hello" --speaker 0
python main.py --text "Hello" --speaker 1
python main.py --text "Hello" --speaker 2
```

### Voice Temperature Control
```bash
# More robotic (lower temperature)
python main.py --text "Hello" --temperature 0.3

# More natural (higher temperature)
python main.py --text "Hello" --temperature 1.5
```

### Batch Processing
```bash
# Process multiple files
for file in *.txt; do
  python main.py --file "$file"
done
```

### Custom Configuration
```bash
# Use custom settings
python main.py --config gaming_config.yaml --interactive
```

### API Integration Patterns

#### Webhook Integration
```python
# Example: Respond to chat messages
@app.route('/chat', methods=['POST'])
def handle_chat():
    message = request.json['message']
    
    # Convert to speech
    tts_response = requests.post('http://localhost:8080/tts', 
                               json={'text': f"Chat message: {message}"})
    
    return {'status': 'spoken'}
```

#### Real-time Integration
```python
# Example: Live chat TTS
import asyncio
import aiohttp

async def speak_message(text):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:8080/tts',
                               json={'text': text}) as response:
            return await response.json()

# Usage
asyncio.run(speak_message("Live message from chat"))
```

---

## 📞 Support & Community

### Getting Help
- **Documentation**: Check `docs/README.md` for technical details
- **Issues**: Report bugs and feature requests
- **Configuration**: See `config/config.yaml` for all settings

### Contributing
- **Bug Reports**: Include error messages and steps to reproduce
- **Feature Requests**: Describe your use case and expected behavior
- **Code Contributions**: Follow the project's coding standards

### Best Practices
1. **Test First**: Use `--test-tts` before important sessions
2. **Backup Config**: Save working configurations
3. **Monitor Performance**: Watch for memory usage in long sessions
4. **Update Regularly**: Keep the application updated

---

## 🎉 Conclusion

The TTS Virtual Microphone application provides powerful text-to-speech capabilities for gaming, streaming, accessibility, and development. Whether you're using the simple CLI commands or building complex API integrations, this guide should help you get the most out of the application.

**Quick Reference:**
- **Simple Use**: `python main.py --text "Hello"`
- **Gaming/Streaming**: `python main.py --interactive`
- **Development**: `python main.py --api`
- **Help**: `python main.py --help`

Happy text-to-speech conversion! 🎤✨

---

*Last updated: January 2025*
*Application Version: 1.0.0*
