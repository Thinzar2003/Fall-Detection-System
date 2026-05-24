# 🛡️ Fall Detection System
### MediaPipe 0.10+ · Acceleration Analysis · Real-time Alert · CSV Logging

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green?logo=google)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red?logo=opencv)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

A real-time fall detection system built with **MediaPipe Pose**, using straight-line geometry and acceleration analysis to detect falls from a webcam or image dataset. Includes sound alert, CSV logging, and live chart visualization.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Concept Pipeline](#concept-pipeline)
- [Equations Used](#equations-used)
- [Features](#features)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dataset](#dataset)
- [Results](#results)
- [Tools & Libraries](#tools--libraries)

---

## Overview

Falls are a leading cause of injury — especially for the elderly — and often go undetected in unsupervised environments. This project uses a **camera-only approach** (no wearables) to detect falls in real time by analyzing body pose landmarks extracted by MediaPipe.

The system applies three mathematical layers:
1. **Straight-line geometry** — compute body angles and positions
2. **Acceleration from frame timestamps** — detect rapid downward motion
3. **Multi-condition fall gate** — combine signals to make a robust decision

---

## Concept Pipeline

```
Webcam / Image
      ↓
MediaPipe Pose  →  33 body landmarks (x, y, z)
      ↓
Layer 1: Geometry
  • Midpoint of shoulders and hips
  • Euclidean distance (torso length)
  • Torso angle from vertical (dot-product)
      ↓
Layer 2: Acceleration  [webcam mode only]
  • Velocity  = Δy / Δt
  • Acceleration = Δvel / Δt
      ↓
Layer 3: Fall Decision Gate
  • Condition A: acceleration > 800 px/s²
  • Condition B: torso angle  > 45°
  • Condition C: hip height   > 65% of frame
  • Fall = A AND B AND C
      ↓
Alert + CSV Log
```

---

## Equations Used

| # | Equation | Formula | Purpose |
|---|---|---|---|
| 1 | Midpoint | `M = (P1 + P2) / 2` | Centre of shoulder/hip pair |
| 2 | Euclidean Distance | `d = √(Δx² + Δy²)` | Torso vector length |
| 3 | Torso Angle | `θ = arccos(v⃗·up⃗ / \|v⃗\|)` | Body tilt from vertical |
| 4 | Velocity | `v(t) = Δy / Δt` | Downward speed of hips |
| 5 | Acceleration | `a(t) = Δv / Δt` | Sudden drop detection |
| 6 | Hip Height Ratio | `R = y_hip / H_frame` | Normalised body position |
| 7 | Fall Decision | `A AND B AND C` | Multi-condition gate |
| 8 | F1 Score | `2PR / (P + R)` | Detection performance |

### Why these equations?

**Equation 3 (Torso Angle)** is the most important — it uses the dot-product formula to measure how far the body has tilted from upright. A person standing has ~0°; a person lying on the ground has ~90°.

**Equation 5 (Acceleration)** captures the *sudden* nature of a real fall. Slow bending or sitting produces low acceleration. Only a genuine fall produces a sharp spike in downward acceleration.

**Equation 7 (AND gate)** prevents false alarms — using OR would trigger on jumping (A only) or bowing (B only). All three conditions together form the unique signature of a fall.

---

## Features

| Feature | Detail |
|---|---|
| 🎥 **Live webcam** | Real-time detection from any camera |
| 🎞️ **Video file** | Process any `.mp4` / `.avi` file |
| 🖼️ **Image dataset** | Test against YOLO-format labelled images |
| 🔊 **Sound alert** | Cross-platform beep on fall detection |
| 📄 **CSV logging** | Every frame logged with all metrics |
| 📊 **Live chart** | Rolling acceleration + angle plot |
| 🏷️ **TP/FP/FN/TN** | Confusion matrix per frame |
| 📈 **Accuracy metrics** | Precision, Recall, F1, Accuracy |

---

## Project Structure

```
fall-detection-system/
│
├── fall_detection_full.ipynb     ← Main notebook (all cells)
├── fall_detection_presentation.pptx  ← Presentation slides
├── pose_landmarker_lite.task     ← MediaPipe model (auto-downloaded)
├── .gitignore
└── README.md
```

### Notebook Structure

```
Cell 1  → Install packages + download MediaPipe model
Cell 2  → All helpers: geometry, tracker, detector, HUD, logger
Cell 3A → LIVE WEBCAM / VIDEO FILE detection
Cell 3B → IMAGE DATASET testing with YOLO labels
Cell 4  → Post-session review chart from CSV
```

---

## How to Run

### Requirements
- Python 3.10+
- Webcam (for Cell 3A)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOURUSERNAME/fall-detection-system.git
cd fall-detection-system

# Create virtual environment
python -m venv falldetect
falldetect\Scripts\activate      # Windows
# source falldetect/bin/activate # macOS/Linux

# Install dependencies
pip install mediapipe opencv-python numpy matplotlib ipywidgets pandas jupyter
```

### Run in VS Code

1. Open `fall_detection_full.ipynb` in VS Code
2. Select kernel → **Python (falldetect)**
3. Run **Cell 1** → installs packages + downloads model
4. Run **Cell 2** → loads all helpers
5. Run **Cell 3A** → webcam detection starts
6. Click **Stop** button or press **Kernel → Interrupt** to stop
7. Run **Cell 4** → view session review chart

### Configuration (top of Cell 3A)

```python
SOURCE       = 'webcam'   # or 'path/to/video.mp4'
CAMERA_INDEX = 0          # 0 = built-in, 1+ = external
DISPLAY_W    = 960        # display width in pixels
```

### Tuning thresholds (Cell 2 — FallDetector class)

```python
ACCEL_THRESHOLD  = 800    # px/s²  — lower = more sensitive
ANGLE_THRESHOLD  = 45     # degrees from vertical
HIP_HEIGHT_RATIO = 0.65   # fraction of frame height
COOLDOWN         = 5.0    # seconds between alerts
```

---

## Dataset

Tested on the **Fall Detection Dataset** from Kaggle (YOLO format):
- 374 training images with `.txt` label files
- Label format: `class x_center y_center width height`
- Class `0` = fall, Class `1` = no fall

To test with your own dataset, set in Cell 3B:
```python
DATASET_ROOT = r'path\to\your\dataset'
SPLIT        = 'train'   # or 'val'
```

---

## Results

After running Cell 3B on the training split:

| Metric | Description |
|---|---|
| **Accuracy** | `(TP + TN) / total` |
| **Precision** | `TP / (TP + FP)` — how many detections were real falls |
| **Recall** | `TP / (TP + FN)` — how many real falls were caught |
| **F1 Score** | `2 × P × R / (P + R)` — balanced metric |

> Note: Static image mode uses geometry only (angle + hip height) since acceleration requires consecutive frames with timestamps.

---

## MediaPipe API (0.10+ vs 0.9)

This project uses the new `mediapipe.tasks` API:

| Old `mp.solutions` (0.9) | New `mediapipe.tasks` (0.10+) |
|---|---|
| `mp.solutions.pose.Pose()` | `PoseLandmarker.create_from_options()` |
| `pose.process(rgb_frame)` | `landmarker.detect_for_video(mp_image, ts)` |
| `results.pose_landmarks.landmark[i]` | `result.pose_landmarks[0][i]` |
| `mp_draw.draw_landmarks()` | Manual `cv2.line` / `cv2.circle` |

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| [MediaPipe](https://developers.google.com/mediapipe) | Body pose landmark detection |
| [OpenCV](https://opencv.org) | Webcam capture and frame annotation |
| [NumPy](https://numpy.org) | Vector math and geometric calculations |
| [Matplotlib](https://matplotlib.org) | Live charts and post-session review |
| [Pandas](https://pandas.pydata.org) | CSV log analysis |
| [ipywidgets](https://ipywidgets.readthedocs.io) | Interactive notebook widgets |
| [Jupyter](https://jupyter.org) | Notebook environment (VS Code) |

---

## Acknowledgements

- Dataset: [Kaggle Fall Detection Dataset](https://www.kaggle.com)
- Pose model: [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

---

*Computer Vision Project — Fall Detection using MediaPipe Pose and Acceleration Analysis*
