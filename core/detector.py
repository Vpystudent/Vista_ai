import os
import time
import base64
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Common synonyms mapping for user friendly manual target entry
SYNONYMS = {
    "drone": ["airplane", "bird"],
    "uav": ["airplane"],
    "phone": ["cell phone"],
    "mobile": ["cell phone"],
    "smartphone": ["cell phone"],
    "bag": ["backpack", "handbag", "suitcase"],
    "human": ["person"],
    "man": ["person"],
    "woman": ["person"],
    "people": ["person"],
    "bike": ["bicycle", "motorcycle"],
    "vehicle": ["car", "truck", "bus", "motorcycle"],
    "automobile": ["car"],
    "auto": ["car"],
}

class VistaDetector:
    """
    VISTA Drone AI Detection Engine.
    Specialized for real-time live feed detection with manual target selection
    and confidence percentage estimation.
    """
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[VISTA] Initializing YOLO11 on device: {self.device}")
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.class_names = self.model.names if hasattr(self.model, "names") else {}

        # Warm up the model so the first user frame does not lag or timeout
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            with torch.inference_mode():
                self.model.predict(source=dummy, imgsz=320, conf=0.35, device=self.device, verbose=False)
            print("[VISTA] YOLO11 warmup complete.")
        except Exception as e:
            print(f"[VISTA Warmup Note]: {e}")

    def resolve_target_classes(self, manual_targets: list) -> set:
        """
        Maps user input strings (e.g. ['drone', 'person', 'phone'])
        to valid YOLO model class names.
        """
        if not manual_targets:
            return set()  # Empty means detect all

        target_set = set()
        for t in manual_targets:
            t_clean = t.strip().lower()
            if not t_clean:
                continue

            # Check exact match
            matched = False
            for c_id, c_name in self.class_names.items():
                if t_clean == c_name.lower():
                    target_set.add(c_name.lower())
                    matched = True

            # Check synonym dictionary
            if not matched and t_clean in SYNONYMS:
                for syn in SYNONYMS[t_clean]:
                    target_set.add(syn.lower())
                matched = True

        return target_set

    def detect_frame_boxes(self, frame_bgr: np.ndarray, manual_targets: list = None, conf: float = 0.35, iou: float = 0.45):
        """
        Infers targets on a raw video frame.
        Filters ONLY the objects requested manually by the user.
        Returns target coordinates, confidence percentage, and latency.
        """
        start_time = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        # Resolve target classes from user input
        active_targets = self.resolve_target_classes(manual_targets)

        # Run YOLO11 inference with imgsz=320 for rapid CPU/GPU real-time throughput
        with torch.inference_mode():
            results = self.model.predict(
                source=frame_bgr,
                imgsz=320,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                raw_name = self.class_names.get(cls_id, f"target_{cls_id}").lower()
                confidence = float(box.conf[0].item())

                # If manual targets are specified, strictly filter out everything else!
                if active_targets and raw_name not in active_targets:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                # Clamp within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)

                conf_percent = round(confidence * 100, 1)

                detections.append({
                    "id": idx + 1,
                    "class_name": raw_name.title(),
                    "confidence": confidence,
                    "confidence_pct": conf_percent,
                    "box": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1
                })

        return {
            "success": True,
            "latency_ms": round(latency_ms, 1),
            "fps": round(fps, 1),
            "total_found": len(detections),
            "manual_targets_active": list(active_targets) if active_targets else ["ALL"],
            "detections": detections,
            "frame_width": width,
            "frame_height": height
        }
