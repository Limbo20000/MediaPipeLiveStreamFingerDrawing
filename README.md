# MediaPipeLiveStreamFingerDrawing

This project implements MediaPipe's hand detection to enable real-time drawing on a blank canvas using hand gestures.
Detect hand larndmark points on image and on live video feed.

## Features

- **Real-Time Hand Detection**: Utilizes MediaPipe's hand detection model to identify and track hand gestures in live video feed.
  
- **Gesture Recognition**: Trained the model to recognize four specific gestures which are: Flat, Palm, Thumbs Up and Circle.

- **Canvas Drawing**: Enables users to draw directly on a blank canvas in real time based on hand gestures.

## Usage

1. **Install Dependencies**: Make sure to install the necessary libraries and dependencies. You can use `pip` to install required packages:
   ```bash
   pip install -r requirements.txt
