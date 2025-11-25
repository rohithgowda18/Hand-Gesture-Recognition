# Copilot Instructions for Hand Gesture Recognition

## Project Overview
Real-time hand gesture recognition system using MediaPipe. Detects hand landmarks via webcam, classifies hand signs (3 base classes: open hand, closed fist, pointing) and finger gestures (4 base classes: stationary, clockwise, counterclockwise, moving) using TFLite MLP models.

## Architecture

### Data Flow
1. **MediaPipe Detection** (`app.py:60-62`): Converts video frames to 21 hand landmarks (3D coords)
2. **Preprocessing**: 
   - `calc_landmark_list()`: Converts landmarks to pixel coordinates
   - `pre_process_landmark()`: Normalizes to relative coords, converts to 1D array (42 values), normalizes by max value
   - `pre_process_point_history()`: Tracks fingertip history as normalized deltas (32 values for 16-frame deque)
3. **Classification**:
   - `KeyPointClassifier`: TFLite inference on hand pose → hand sign ID
   - `PointHistoryClassifier`: TFLite inference on fingertip trajectory → gesture ID
4. **Visualization**: Draws landmarks, bounding boxes, and text overlays

### Core Components
- `app.py`: Main inference loop (webcam capture, detection, classification, rendering)
- `model/keypoint_classifier/`: Hand sign recognition (models + label CSVs)
- `model/point_history_classifier/`: Finger gesture recognition (models + label CSVs)
- `utils/CvFpsCalc`: FPS measurement utility using OpenCV tick counters

## Key Patterns

### Model Inference Pattern
Both classifiers follow identical TFLite pattern (`model/keypoint_classifier/keypoint_classifier.py`):
```python
interpreter.set_tensor(input_index, np.array([data], dtype=np.float32))
interpreter.invoke()
result_index = np.argmax(interpreter.get_tensor(output_index))
```
`PointHistoryClassifier` adds confidence thresholding via `score_th` parameter.

### Data Collection Workflow
- Press `k` → collect keypoints → press `0-9` → appends to `model/keypoint_classifier/keypoint.csv`
- Press `h` → collect point history → press `0-9` → appends to `model/point_history_classifier/point_history.csv`
- CSV format: `class_id, feature1, feature2, ..., featureN`

### Training Notebooks
Both notebooks follow identical pattern:
1. Load CSV data → split train/test
2. Define MLP model (optional LSTM for gestures with `use_lstm=True`)
3. Train → evaluate → convert to TFLite → save with updated label CSV

**Critical**: Update `NUM_CLASSES` constant when adding/removing gesture types.

## Development Workflows

### Run Demo
```bash
python app.py --device 0 --width 960 --height 540 --min_detection_confidence 0.7
```

### Collect Training Data
1. Run `app.py` normally
2. Press `k` (keypoint mode) or `h` (history mode)
3. For each gesture: press gesture number (0-9), perform gesture, release
4. Data accumulates in respective CSVs

### Train Models
1. Open `keypoint_classification.ipynb` or `point_history_classification.ipynb` in Jupyter
2. Execute cells top-to-bottom
3. Models overwrite `.tflite` files; update label CSVs manually if classes changed

### File Dependencies
- `.csv` files are source of truth for training data (NOT in version control typically)
- `.tflite` files are pre-trained models (inference-only, immutable at runtime)
- `.hdf5` files are Keras checkpoints (for training resumption only)
- `*_label.csv`: Maps class IDs to human-readable labels (1 label per line)

## Important Conventions

### Preprocessing Details
- **Keypoint normalization**: Relative to wrist (landmark[0]), all landmarks normalized by max distance
- **Point history**: 16-frame deque of fingertip coords, normalized by image dimensions
- **Anomaly handling**: Zero values `[0, 0]` in point history indicate no active pointing gesture

### Configuration Parameters (app.py)
- `--use_static_image_mode`: Disables tracking (slower, more stable for still images)
- `--min_detection_confidence`: Threshold for hand detection (default 0.7)
- `--min_tracking_confidence`: Threshold for hand tracking between frames (default 0.5)
- `history_length=16`: Gesture recognition buffer (hardcoded, tied to model training)

### Class Structure
- Keypoint classifier: 0=Open, 1=Closed, 2=Pointing (extendable to 10 classes)
- Point history classifier: 0=Stationary, 1=Clockwise, 2=Counterclockwise, 4=Moving (class ID 3 skipped)

## Integration Points

### MediaPipe Hands API
- `static_image_mode`: Inference mode toggle
- `max_num_hands=1`: Hardcoded single-hand detection
- `results.multi_hand_landmarks`: List of 21-point hand meshes
- `results.multi_handedness`: Hand labels (Left/Right) with confidence scores

### TensorFlow Lite
- Models expect `np.float32` input tensors with shape `(1, feature_count)`
- Output is unnormalized logits; use `argmax` for classification
- Thread count hardcoded to 1 for deterministic results

### CSV I/O
- `logging_csv()`: Appends rows in real-time during data collection
- Label CSVs: Plain text, one label per line (no quotes, no headers)

## Common Modifications

**Add new gesture class**:
1. Collect data: Run app, press `h`, record new gesture as class ID 5+
2. Update label CSV: Add line to `model/point_history_classifier/point_history_classifier_label.csv`
3. Retrain: Open notebook, set `NUM_CLASSES=5`, run all cells
4. Verify: Check new `.tflite` model loads without shape errors

**Change input resolution**:
- Modify `pre_process_point_history()` to account for new image dimensions (uses `image_width`, `image_height`)
- Rerun data collection and retraining if accuracy drops

**Multi-hand detection**:
- Change `max_num_hands` in MediaPipe config
- Requires loop refactoring in `app.py` (currently assumes single hand)

## External Dependencies
- **mediapipe 0.8.1+**: Hand pose estimation
- **opencv (cv2) 3.4.2+**: Video capture, drawing
- **tensorflow 2.3.0+**: TFLite model loading
- **scikit-learn, matplotlib**: Training only (confusion matrices, model evaluation)
