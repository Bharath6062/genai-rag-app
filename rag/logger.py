# rag/logger.py
import json
from pathlib import Path
from datetime import datetime

def write_run_log(logs_dir: str, payload: dict) -> str:
    """
    Writes one JSON log file per run.
    Returns the saved file path.
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(logs_dir) / f"run_{ts}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(path)
