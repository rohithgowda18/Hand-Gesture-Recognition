#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-mode Hand Gesture Recognition and Air Canvas Application
Combines gesture recognition with air canvas drawing functionality
"""

import csv
import copy
import argparse
import itertools
from collections import Counter, deque, defaultdict
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import colorchooser

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


# ==================== COLOR PICKER ====================
def show_color_picker(current_color):
    """Opens a color picker window and returns the selected color in BGR format"""
    root = tk.Tk()
    root.withdraw()
    rgb_color = (current_color[2], current_color[1], current_color[0])
    color_code = colorchooser.askcolor(
        title="Choose drawing color",
        initialcolor=rgb_color
    )
    if color_code[0] is None:
        return None
    selected_rgb = color_code[0]
    selected_bgr = (int(selected_rgb[2]), int(selected_rgb[1]), int(selected_rgb[0]))
    return selected_bgr


# ==================== ARGUMENT PARSING ====================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    args = parser.parse_args()
    return args


# ==================== GESTURE RECOGNITION FUNCTIONS ====================
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
        for index, landmark in enumerate(landmark_point):
            if index in [0, 1, 4, 8, 12, 16, 20]:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            else:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


# ==================== AIR CANVAS FUNCTIONS ====================
def draw_canvas_buttons(img, current_color, current_brush_size):
    """Draw UI buttons for canvas mode"""
    cv.rectangle(img, (20, 10), (140, 60), (50, 50, 50), -1)
    cv.putText(img, "CLEAR", (40, 45), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.rectangle(img, (160, 10), (260, 60), current_color, -1)
    text_color = (255, 255, 255) if sum(current_color) < 382 else (0, 0, 0)
    cv.putText(img, "COLOR", (175, 45), cv.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv.rectangle(img, (500, 10), (620, 60), (100, 100, 100), -1)
    cv.putText(img, "SAVE", (525, 45), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.putText(img, f"Size: {current_brush_size}", (280, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)


def save_canvas(persistent_canvas):
    """Save the drawing with timestamp"""
    filename = f"draw/assets/air_canvas_{int(time.time())}.png"
    cv.imwrite(filename, persistent_canvas)
    print(f"Canvas saved as {filename}")


def draw_mode_info(image, app_mode):
    """Display current mode on the image"""
    mode_text = "MODE: GESTURE RECOGNITION" if app_mode == "gesture" else "MODE: AIR CANVAS"
    cv.putText(image, mode_text, (10, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(image, "Press 'g' for Gesture | 'd' for Draw | 'esc' to exit", (10, image.shape[0] - 50),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ==================== MAIN APPLICATION ====================
def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Gesture recognition state
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0

    # Canvas state
    deque_len = 1024
    drawing_enabled = False
    current_color = (0, 0, 255)
    current_brush_size = 2
    color_points = defaultdict(lambda: deque(maxlen=deque_len))
    last_button_time = time.time()
    cooldown_time = 0.5
    prev_x, prev_y = 0, 0
    smoothing = 0.3
    persistent_canvas = np.ones((args.height, args.width, 3), dtype=np.uint8) * 255

    # Application mode
    app_mode = "gesture"  # "gesture" or "draw"
    use_brect = True

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)

        # Mode switching
        if key == 27:  # ESC
            break
        elif key == ord('g'):
            app_mode = "gesture"
            print("Switched to Gesture Recognition Mode")
        elif key == ord('d'):
            app_mode = "draw"
            print("Switched to Air Canvas Mode")

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if app_mode == "gesture":
            # ==================== GESTURE RECOGNITION MODE ====================
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (history_length * 2):
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()

                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image, brect, handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, -1)

            # Handle data collection in gesture mode
            number = -1
            if 48 <= key <= 57:
                number = key - 48
            if key == 110:
                mode = 0
            if key == 107:
                mode = 1
            if key == 104:
                mode = 2

        else:
            # ==================== AIR CANVAS MODE ====================
            temp_canvas = persistent_canvas.copy()
            draw_canvas_buttons(debug_image, current_color, current_brush_size)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    x, y = int(index_tip.x * args.width), int(index_tip.y * args.height)

                    smoothed_x = int(prev_x + smoothing * (x - prev_x))
                    smoothed_y = int(prev_y + smoothing * (y - prev_y))
                    prev_x, prev_y = smoothed_x, smoothed_y
                    smoothed = (smoothed_x, smoothed_y)

                    cv.circle(debug_image, smoothed, 8, (0, 0, 0), -1)

                    thumb_x, thumb_y = int(thumb_tip.x * args.width), int(thumb_tip.y * args.height)
                    pinch_distance = np.hypot(thumb_x - x, thumb_y - y)
                    drawing_enabled = pinch_distance >= 40

                    if smoothed_y < 65:
                        now = time.time()
                        if now - last_button_time > cooldown_time:
                            if 20 <= smoothed_x <= 140:
                                color_points.clear()
                                persistent_canvas.fill(255)
                            elif 160 <= smoothed_x <= 260:
                                new_color = show_color_picker(current_color)
                                if new_color:
                                    current_color = new_color
                            elif 500 <= smoothed_x <= 620:
                                save_canvas(persistent_canvas)
                            last_button_time = now
                    elif drawing_enabled:
                        color_points[current_color].appendleft(smoothed)
                    else:
                        color_points[current_color].appendleft(None)
            else:
                color_points[current_color].appendleft(None)

            # Draw lines for all colors
            for color, points in color_points.items():
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    cv.line(debug_image, points[i - 1], points[i], color, current_brush_size)
                    cv.line(temp_canvas, points[i - 1], points[i], color, current_brush_size)
                    cv.line(persistent_canvas, points[i - 1], points[i], color, current_brush_size)

            # Handle canvas keyboard input
            if key == ord('+') or key == ord('='):
                current_brush_size = min(20, current_brush_size + 1)
            elif key == ord('-'):
                current_brush_size = max(1, current_brush_size - 1)

            debug_image = temp_canvas

        # Display mode info
        draw_mode_info(debug_image, app_mode)

        cv.imshow('Hand Gesture & Canvas Combined', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
