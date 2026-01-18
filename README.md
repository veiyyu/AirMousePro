# AirMouse Pro

AirMouse Pro is a hands-free cursor control system for macOS that lets you control your computer using hand gestures and a webcam.

## What it does
- Move cursor with your index finger  
- Pinch to click  
- Hold pinch to drag  
- Two-finger scroll  
- Two-finger pinch to double-click  
- Hold a closed fist to minimize the current window  
- On-screen cursor overlay  

## Requirements
- macOS  
- Python 3.9+  
- Webcam  

## Setup
```bash
git clone https://github.com/veiyyu/AirMousePro.git
cd AirMousePro
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe pyobjc numpy
python air_mouse.py

## Permissions

- Enable Accessibility and Screen Recording for Terminal or your IDE:
  System Settings → Privacy & Security

## Controls

- Pause / Resume: Cmd + Ctrl + Option + P

## Notes

- Works best with good lighting

- Gesture sensitivity can be adjusted in the config section
