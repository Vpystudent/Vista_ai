# VISTA (Visual Intelligence and Spatial Target Analysis)
> Real-Time Aerial AI Perception System for Drones

---

## 🚁 Overview
VISTA is an aerospace-oriented computer vision platform engineered to detect, classify, and track small and distant targets from aerial drone video.

### Core Modules:
- **Detection Engine:** YOLO11 fine-tuned for high-altitude aerial perspectives.
- **Tracking (Planned):** ByteTrack / BoT-SORT for persistent cross-frame track IDs.
- **Active Learning:** Human-in-the-loop verification pipeline for edge-case retraining.
- **Frontend:** Tactical Drone Mission Control UI, built to be compatible with **Google Stitch** (`stitch.withgoogle.com`).
- **Backend:** High-throughput async REST & MJPEG streaming powered by FastAPI and OpenCV.

---

## ⚡ Quick Start

### 1. Run the Web Dashboard
```bash
python app.py
```
Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🎨 Integrating with Google Stitch (`stitch.withgoogle.com`)

The frontend was designed using the exact semantic HTML and Tailwind CSS pattern that **Google Stitch** exports.

### To customize the UI in Google Stitch:
1. Open **[stitch.withgoogle.com](https://stitch.withgoogle.com)**.
2. Use the provided VISTA prompt to design or customize your dashboard screen.
3. Download/Export the HTML + Tailwind CSS.
4. Replace `templates/index.html` with your exported Stitch HTML.
5. Ensure these key element IDs remain intact for automatic JavaScript binding:
   - `image-file-input` / `video-file-input` (Upload inputs)
   - `annotated-image` (Image result canvas)
   - `conf-slider` / `iou-slider` (Threshold sliders)
   - `stat-total-targets` / `stat-latency` (Telemetry counters)
   - `detection-feed` (Target cards & human feedback container)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Tactical Mission Control Dashboard |
| `GET` | `/api/status` | Hardware & GPU telemetry |
| `POST` | `/api/detect/image` | Runs YOLO11 inference on high-res aerial image |
| `POST` | `/api/detect/video` | Processes uploaded drone video sequence |
| `GET` | `/api/stream/live` | Low-latency MJPEG video stream (webcam / RTSP) |
| `POST` | `/api/feedback` | Records human confirmation or rejection of target |
| `GET` | `/api/feedback/stats`| Cumulative human-in-the-loop review statistics |

---

## 🎯 Fine-Tuning YOLO11 on Drone Datasets (VisDrone / UAVDT)
When you are ready to fine-tune on aerial data:
```python
from ultralytics import YOLO

# Load base model
model = YOLO("yolo11n.pt")

# Train on drone data (e.g., VisDrone.yaml)
model.train(
    data="VisDrone.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda:0"
)
```
After training, copy `runs/detect/train/weights/best.pt` to `core/best_drone.pt` and switch models in `core/detector.py`.
