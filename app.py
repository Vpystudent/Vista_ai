import os

# Crucial for low-resource cloud containers (Render, HuggingFace, Docker)
# Prevents GNU OpenMP / MKL multi-threading deadlocks and RAM exhaustion
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"

import json
import base64
import time
import cv2
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_grad_enabled(False)

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from core.detector import VistaDetector
from core.feedback import HumanVerificationManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app = FastAPI(title="VISTA Live Drone Feed & Target Acquisition", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize AI Detector & Human Feedback Manager
detector = VistaDetector("yolo11n.pt")
feedback_mgr = HumanVerificationManager()

@app.get("/")
async def serve_home():
    """Serves the VISTA Live Drone Feed HUD."""
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/api/status")
async def get_system_status():
    """Returns runtime telemetry and model class list."""
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (Host)"
    return {
        "status": "ONLINE",
        "device": device_name,
        "cuda": gpu_available,
        "active_model": detector.model_path,
        "supported_classes": list(detector.class_names.values()),
        "feedback_stats": feedback_mgr.get_stats()
    }

@app.post("/api/detect/frame")
async def detect_frame_http(request: Request):
    """
    HTTP fallback endpoint for single frame detection.
    Runs inference in threadpool so the asyncio event loop is never blocked.
    """
    try:
        body = await request.json()
        img_b64 = body.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        targets = body.get("targets", [])
        conf = float(body.get("conf", 0.35))
        iou = float(body.get("iou", 0.45))

        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"success": False, "error": "Invalid frame"})

        res = await run_in_threadpool(
            detector.detect_frame_boxes, frame, targets, conf, iou
        )
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """
    Ultra-low-latency bidirectional WebSocket connection.
    Client streams camera frames -> Server runs YOLO11 with manual target filtering
    -> Returns bounding box coordinates and confidence percentages.
    """
    await websocket.accept()
    try:
        while True:
            text_data = await websocket.receive_text()
            payload = json.loads(text_data)

            img_b64 = payload.get("image", "")
            if not img_b64:
                continue

            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]

            targets = payload.get("targets", [])
            conf = float(payload.get("conf", 0.35))
            iou = float(payload.get("iou", 0.45))

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                result = await run_in_threadpool(
                    detector.detect_frame_boxes, frame, targets, conf, iou
                )
                await websocket.send_text(json.dumps(result))
            else:
                await websocket.send_text(json.dumps({"success": False, "detections": []}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Error]: {e}")

@app.post("/api/feedback")
async def record_feedback_endpoint(request: Request):
    """Records human-in-the-loop confirmation or rejection of detected target."""
    data = await request.json()
    return feedback_mgr.record_feedback(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
