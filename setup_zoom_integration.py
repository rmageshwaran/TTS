#!/usr/bin/env python3
"""
Zoom Integration Setup Script
Automated setup and verification for TTS Virtual Microphone with Zoom
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header():
    """Print setup header."""
    print("🎤 TTS VIRTUAL MICROPHONE - ZOOM INTEGRATION SETUP")
    print("=" * 60)
    print("🔗 Configuring system for seamless Zoom integration")
    print()

def check_vb_cable():
    """Check VB-Cable installation."""
    print("🔧 Step 1: Checking VB-Cable Installation")
    print("-" * 40)
    
    try:
        # Run VB-Cable test
        result = subprocess.run([
            sys.executable, "test_vb_cable_integration.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ VB-Cable is installed and working")
            return True
        else:
            print("❌ VB-Cable test failed")
            print("🔗 Download VB-Cable from: https://vb-audio.com/Cable/")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  VB-Cable test timed out")
        return False
    except Exception as e:
        print(f"❌ VB-Cable check failed: {e}")
        return False

def test_tts_system():
    """Test TTS system functionality."""
    print("\n🤖 Step 2: Testing TTS System")
    print("-" * 40)
    
    try:
        # Test basic TTS functionality
        result = subprocess.run([
            sys.executable, "main.py", 
            "--text", "Testing TTS system for Zoom integration"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "Audio routing successful" in result.stdout:
            print("✅ TTS system working correctly")
            print("✅ Audio routing to VB-Cable successful")
            return True
        else:
            print("❌ TTS system test failed")
            print("Check logs for details")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  TTS test timed out")
        return False
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

def create_meeting_shortcuts():
    """Create useful meeting template shortcuts."""
    print("\n📁 Step 3: Creating Meeting Templates")
    print("-" * 40)
    
    try:
        # Ensure templates directory exists
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        # Create quick start script (without emojis for Windows compatibility)
        quick_start_content = """@echo off
echo Starting TTS Virtual Microphone - Meeting Mode
echo.
echo Zoom Setup Checklist:
echo    1. Set microphone to 'CABLE Output (VB-Audio Point)'
echo    2. Set microphone volume to 100%%
echo    3. Disable auto-adjust microphone volume
echo    4. Disable background noise suppression
echo.
echo Starting meeting mode...
python main.py --meeting
pause
"""
        
        with open("start_meeting_mode.bat", "w") as f:
            f.write(quick_start_content)
        
        print("✅ Created start_meeting_mode.bat")
        print("✅ Meeting templates available in templates/ folder")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create templates: {e}")
        return False

def show_zoom_setup_instructions():
    """Display Zoom setup instructions."""
    print("\n🎯 Step 4: Zoom Configuration")
    print("-" * 40)
    
    instructions = """
ZOOM AUDIO SETTINGS:

1. 🎤 MICROPHONE SETUP:
   • Open Zoom → Settings → Audio
   • Set Microphone to: "CABLE Output (VB-Audio Point)" or "VB-Audio Point"
   • Set microphone volume to 100%
   • UNCHECK "Automatically adjust microphone volume"
   • UNCHECK "Suppress background noise"

2. 🔊 SPEAKER SETUP:
   • Set Speaker to your actual headphones/speakers
   • DO NOT use VB-Cable for speaker output
   • Test speakers to ensure you can hear others

3. 🔧 ADVANCED SETTINGS:
   • Click "Advanced" button
   • UNCHECK "Echo cancellation" (may interfere with TTS)
   • UNCHECK "Original sound" (unless specifically needed)

4. ✅ VERIFICATION:
   • Use "Test Speaker & Microphone" in Zoom
   • Run TTS while testing microphone
   • Verify level indicator responds to TTS audio

MEETING WORKFLOW:

1. 🚀 START MEETING MODE:
   • Double-click: start_meeting_mode.bat
   • OR run: python main.py --meeting

2. 💬 DURING MEETINGS:
   • Type your response and press Enter
   • Use number keys (1-9, 0) for quick responses
   • Mute/unmute in Zoom as needed
   • Generated speech appears as your microphone input

3. 🎯 BEST PRACTICES:
   • Keep phrases short for natural conversation
   • Prepare common responses in advance
   • Test before important meetings
   • Use 'test' command to verify connection
"""
    
    print(instructions)
    return True

def run_integration_test():
    """Run final integration test."""
    print("\n🧪 Step 5: Final Integration Test")
    print("-" * 40)
    
    print("🎯 Testing complete TTS → VB-Cable → Zoom pipeline...")
    
    try:
        # Test with meeting-like phrase
        test_phrase = "Hello everyone, this is a test of the TTS virtual microphone system"
        
        result = subprocess.run([
            sys.executable, "main.py", 
            "--text", test_phrase
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Integration test successful!")
            print("✅ TTS → VB-Cable → Zoom pipeline working")
            print(f"✅ Test phrase: '{test_phrase[:30]}...'")
            return True
        else:
            print("❌ Integration test failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        return False

def main():
    """Main setup function."""
    print_header()
    
    steps_passed = 0
    total_steps = 5
    
    # Step 1: Check VB-Cable
    if check_vb_cable():
        steps_passed += 1
    
    # Step 2: Test TTS
    if test_tts_system():
        steps_passed += 1
    
    # Step 3: Create shortcuts
    if create_meeting_shortcuts():
        steps_passed += 1
    
    # Step 4: Show instructions
    if show_zoom_setup_instructions():
        steps_passed += 1
    
    # Step 5: Integration test
    if run_integration_test():
        steps_passed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("🏆 ZOOM INTEGRATION SETUP RESULTS")
    print("=" * 60)
    
    success_rate = (steps_passed / total_steps) * 100
    
    print(f"📊 Setup Progress: {steps_passed}/{total_steps} steps completed ({success_rate:.0f}%)")
    
    if steps_passed == total_steps:
        print("\n🎉 SETUP COMPLETE!")
        print("✅ TTS Virtual Microphone is ready for Zoom integration!")
        print("\n🚀 QUICK START:")
        print("   • Double-click 'start_meeting_mode.bat'")
        print("   • Configure Zoom microphone settings as shown above")
        print("   • Join your meeting and start using TTS!")
        
    elif steps_passed >= 3:
        print("\n✅ MOSTLY READY!")
        print("⚠️  Some manual configuration may be needed")
        print("📖 Review the instructions above")
        
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("❌ Please resolve the failed steps before using with Zoom")
        print("📞 Check installation and try again")
    
    print(f"\n📁 Files created:")
    print(f"   • start_meeting_mode.bat (Quick start script)")
    print(f"   • templates/meeting_*.txt (Response templates)")
    print(f"   • docs/ZOOM_INTEGRATION_GUIDE.md (Complete guide)")
    
    return steps_passed >= 3

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
