# -*- coding: utf-8 -*-
import json
import glob
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/root/stockbot")
HIST = ROOT / "logs/metrics_history.jsonl"


def latest_meta():
    files = sorted(
        glob.glob(str(ROOT / "core/models/xgb_spike4_v*.meta.json")),
        key=os.path.getmtime,
    )
    if not files:
        return None
    return Path(files[-1])


def main():
    p = latest_meta()
    if not p:
        return
    meta = json.loads(p.read_text())
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "date_max": meta.get("date_max"),
        "auc": meta.get("auc_valid"),
        "ap": meta.get("ap_valid"),
        "iters": meta.get("best_iteration"),
        "features": meta.get("features"),
        "target": meta.get("target"),
        "train_rows": meta.get("train_rows"),
        "valid_rows": meta.get("valid_rows"),
        "meta_file": str(p),
    }
    with open(HIST, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
