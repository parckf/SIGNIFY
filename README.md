# Sign Language to Text

Real-time ASL recognition from a webcam feed using MediaPipe hand landmarks and a Random Forest classifier.

## Features

- **Real-time hand tracking** via MediaPipe (21 landmarks)
- **Gesture classification** with scikit-learn Random Forest
- **Sentence assembly** with a 7-frame hold threshold
- **WebSocket chat** for two-way communication with a hearing counterpart

## Prerequisites

- Python 3.7+
- Webcam
- (Optional) Arduino Uno for buzzer/haptic feedback

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt   
