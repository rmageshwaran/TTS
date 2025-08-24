@echo off
echo Starting TTS Virtual Microphone - Meeting Mode
echo.
echo Zoom Setup Checklist:
echo    1. Set microphone to CABLE Output (VB-Audio Point)
echo    2. Set microphone volume to 100%%
echo    3. Disable auto-adjust microphone volume
echo    4. Disable background noise suppression
echo.
echo Starting meeting mode...
python main.py --meeting
pause
