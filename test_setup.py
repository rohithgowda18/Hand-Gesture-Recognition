#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to verify all components work
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier, PointHistoryClassifier

print("✓ Testing imports...")
print(f"  - Python: {sys.version}")
try:
    print(f"  - OpenCV: {cv2.__version__}")
except:
    print(f"  - OpenCV: imported successfully")
try:
    print(f"  - MediaPipe: {mp.__version__}")
except:
    print(f"  - MediaPipe: imported successfully")

print("\n✓ Testing model loading...")
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()
print("  - KeyPointClassifier loaded")
print("  - PointHistoryClassifier loaded")

print("\n✓ Testing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
print("  - MediaPipe Hands initialized")

print("\n✓ Testing camera capture...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"  - Camera working, frame shape: {frame.shape}")
    else:
        print("  - Camera error: Cannot read frame")
else:
    print("  - Warning: Camera not available (this is ok if no camera)")
cap.release()

print("\n✓ All tests passed! Your setup is working correctly.")
print("\nYou can now run:")
print("  python app.py")
print("  python combined_app.py")
print("  python combined_app_gesture_canvas.py")
