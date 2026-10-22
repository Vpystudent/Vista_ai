import json
import os
import time
from typing import Dict, Any

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "..", "feedback_data")
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "human_verification_log.jsonl")

os.makedirs(FEEDBACK_DIR, exist_ok=True)

class HumanVerificationManager:
    """
    Manages human-in-the-loop feedback.
    Allows aerial operators to confirm or reject detected targets,
    storing high-value edge cases for active learning and model fine-tuning.
    """
    def __init__(self, log_path: str = FEEDBACK_FILE):
        self.log_path = log_path

    def record_feedback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_id": data.get("target_id"),
            "class_name": data.get("class_name"),
            "confidence": data.get("confidence"),
            "status": data.get("status"),  # 'verified' or 'flagged' (false positive)
            "user_notes": data.get("notes", ""),
            "bbox": data.get("bbox", [])
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return {"success": True, "message": f"Target #{data.get('target_id')} recorded as {data.get('status')}."}

    def get_stats(self) -> Dict[str, int]:
        if not os.path.exists(self.log_path):
            return {"total_reviewed": 0, "verified": 0, "flagged": 0}

        total = 0
        verified = 0
        flagged = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    total += 1
                    if item.get("status") == "verified":
                        verified += 1
                    elif item.get("status") == "flagged":
                        flagged += 1
                except Exception:
                    pass

        return {"total_reviewed": total, "verified": verified, "flagged": flagged}
