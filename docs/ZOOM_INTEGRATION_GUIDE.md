# 🎤 Zoom Integration Guide for TTS Virtual Microphone

## Overview
This guide shows how to integrate the TTS Virtual Microphone system with Zoom meetings, allowing you to use AI-generated speech as your microphone input.

## Prerequisites
- ✅ VB-Audio Virtual Cable installed and working
- ✅ TTS Virtual Microphone system operational
- ✅ Zoom desktop client installed

## Step-by-Step Integration

### 1. 🎛️ Verify VB-Cable Setup

First, ensure VB-Cable is working:
```bash
# Test the TTS system
python main.py --file example_script.txt

# Verify you see: "VB-Cable routing successful using Windows WASAPI"
```

### 2. 🎤 Configure Zoom Audio Settings

#### A. Open Zoom Audio Settings
1. Open Zoom desktop application
2. Click your profile picture (top-right)
3. Select **Settings**
4. Navigate to **Audio** tab

#### B. Set Microphone Input
1. In the **Microphone** dropdown, select:
   - **"CABLE Output (VB-Audio Point)"** or
   - **"VB-Audio Point"** 
2. Test your microphone - you should see the level indicator responding

#### C. Disable Auto-Adjust (Important!)
1. **Uncheck** "Automatically adjust microphone volume"
2. **Uncheck** "Suppress background noise" 
3. Set microphone volume to **100%**

#### D. Advanced Settings (Optional)
1. Click **Advanced** button
2. **Uncheck** "Echo cancellation" (may interfere with generated audio)
3. **Uncheck** "Original sound" unless specifically needed

### 3. 🔊 Configure System Audio (Optional)

If you want to hear other participants while using TTS:

#### A. Set Zoom Speaker Output
1. In Zoom Audio settings, set **Speaker** to your actual speakers/headphones
2. Do NOT use VB-Cable for speaker output

#### B. Test Audio Configuration
1. Join a test meeting or use "Test Speaker & Microphone"
2. Verify you can hear Zoom's test sounds
3. Verify your microphone input shows activity when TTS is running

### 4. 🎯 Live TTS Integration

#### A. Prepare Your TTS Text
Create text files with your responses/presentations:
```
# responses.txt
Hello everyone, thank you for joining today's meeting.
I'll be presenting our quarterly results.
Let me share my screen now.
```

#### B. During the Meeting
1. **Start the meeting** with microphone muted initially
2. **Run TTS commands** as needed:
   ```bash
   # Real-time text input
   python main.py --text "Hello everyone, I'm ready to begin the presentation"
   
   # From prepared file
   python main.py --file responses.txt
   
   # Interactive mode
   python main.py --interactive
   ```
3. **Unmute in Zoom** when you want the TTS audio to be heard

#### C. Best Practices During Meetings
- **Mute between TTS inputs** to avoid background noise
- **Prepare common responses** in text files for quick access
- **Use short phrases** for more natural conversation flow
- **Test volume levels** before important meetings

### 5. 🔧 Troubleshooting

#### Issue: Participants can't hear TTS audio
**Solutions:**
- Verify VB-Cable is selected as microphone in Zoom
- Check microphone volume is at 100% in Zoom
- Ensure TTS system shows "VB-Cable routing successful"
- Test with Zoom's "Test Speaker & Microphone" feature

#### Issue: Audio quality is poor
**Solutions:**
- Disable Zoom's noise suppression
- Disable automatic microphone adjustment
- Increase TTS audio volume in system
- Use shorter text phrases for better audio quality

#### Issue: Echo or feedback
**Solutions:**
- Use headphones instead of speakers
- Disable Zoom's echo cancellation
- Ensure speaker output is NOT set to VB-Cable

#### Issue: TTS audio cuts off
**Solutions:**
- Wait for TTS generation to complete before unmuting
- Use `--debug` flag to monitor TTS status
- Increase buffer sizes in config if needed

### 6. 📋 Pre-Meeting Checklist

Before joining important meetings:

- [ ] VB-Cable is working (`python test_vb_cable_integration.py`)
- [ ] Zoom microphone set to "CABLE Output (VB-Audio Point)"
- [ ] Zoom microphone volume at 100%
- [ ] Auto-adjust and noise suppression disabled
- [ ] TTS text prepared and tested
- [ ] Audio levels tested in Zoom

### 7. 🎭 Advanced Usage

#### A. Interactive Mode
For real-time conversation:
```bash
python main.py --interactive
# Type your responses and press Enter to generate speech
```

#### B. Hotkey Integration (Future Enhancement)
Consider setting up system hotkeys to trigger common TTS phrases:
- F1: "Thank you"
- F2: "I agree"
- F3: "Let me check on that"
- F4: "Could you repeat that?"

#### C. Meeting Templates
Create template files for different meeting types:
```
# daily_standup.txt
Good morning everyone.
Yesterday I worked on [project name].
Today I plan to focus on [task].
I don't have any blockers.

# presentation_start.txt
Good morning everyone.
Thank you for joining today's presentation.
I'll be covering [topic] today.
Let me share my screen.
```

### 8. 🛡️ Privacy and Ethics

#### Important Considerations:
- **Inform participants** if using AI-generated voice is required by your organization
- **Respect meeting policies** regarding recording and voice modification
- **Use responsibly** - ensure your use case is appropriate
- **Maintain authenticity** in professional communications

### 9. 🔄 Workflow Example

Typical meeting workflow:
1. **Join meeting muted**
2. **Run TTS for greeting**: `python main.py --text "Good morning everyone"`
3. **Unmute briefly** to transmit greeting
4. **Mute again** and prepare next response
5. **Continue pattern** throughout meeting
6. **Use prepared files** for longer responses

## 🎉 Success!

You now have a fully integrated TTS Virtual Microphone system working with Zoom! The AI-generated speech will appear as your microphone input, allowing for seamless meeting participation.

## 📞 Support

If you encounter issues:
1. Run diagnostics: `python test_vb_cable_integration.py`
2. Check logs with: `python main.py --debug`
3. Verify VB-Cable installation in Windows Sound settings
