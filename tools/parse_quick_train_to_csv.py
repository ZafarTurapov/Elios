#!/usr/bin/env python3
import os
import re
import csv
import subprocess
import datetime
import pathlib
import time

CSV_PATH = os.environ.get("METRICS_CSV", "/root/stockbot/logs/training_metrics_v2.csv")
UNIT = os.environ.get("SYSTEMD_UNIT", "quick_train.service")
INVOC = os.environ.get("INVOCATION_ID", "")
TAG = "quick-train"

time.sleep(1.0)  # дадим journald дологировать все строки из quick_train.py


def grab(cmd):
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, text=True, check=False
        ).stdout.splitlines()
    except Exception:
        return []


lines = []
if INVOC:
    lines = grab(
        ["journalctl", "-t", TAG, "-o", "cat", f"_SYSTEMD_INVOCATION_ID={INVOC}"]
    )
if not lines:
    lines = grab(
        [
            "journalctl",
            "-u",
            UNIT,
            "-o",
            "cat",
            "-n",
            "500",
            "--since",
            "10 minutes ago",
        ]
    )
if not lines:
    lines = grab(["journalctl", "-t", TAG, "-o", "cat", "-n", "500"])

pat = {
    "loaded": re.compile(r"Loaded rows:\s*(\d+)"),
    "downsample": re.compile(r"After downsample rows:\s*(\d+)"),
    "shape": re.compile(r"Feature matrix shape:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
    "acc": re.compile(r"Accuracy mean/std:\s*([\d.]+)\s*/\s*([\d.]+)"),
    "prec": re.compile(r"Precision mean/std:\s*([\d.]+)\s*/\s*([\d.]+)"),
    "rec": re.compile(r"Recall mean/std:\s*([\d.]+)\s*/\s*([\d.]+)"),
    "f1": re.compile(r"F1 mean/std:\s*([\d.]+)\s*/\s*([\d.]+)"),
    "saved": re.compile(r"Saved model to\s*(\S+)"),
}

vals = {
    k: None
    for k in [
        "loaded",
        "downsample",
        "n_features",
        "acc_m",
        "acc_s",
        "prec_m",
        "prec_s",
        "rec_m",
        "rec_s",
        "f1_m",
        "f1_s",
        "model_path",
    ]
}

for line in lines:
    m = pat["loaded"].search(line)
    vals["loaded"] = int(m.group(1)) if m else vals["loaded"]
    m = pat["downsample"].search(line)
    vals["downsample"] = int(m.group(1)) if m else vals["downsample"]
    m = pat["shape"].search(line)
    vals["n_features"] = int(m.group(2)) if m else vals["n_features"]
    m = pat["acc"].search(line)
    (
        (
            vals.__setitem__("acc_m", float(m.group(1))),
            vals.__setitem__("acc_s", float(m.group(2))),
        )
        if m
        else None
    )
    m = pat["prec"].search(line)
    (
        (
            vals.__setitem__("prec_m", float(m.group(1))),
            vals.__setitem__("prec_s", float(m.group(2))),
        )
        if m
        else None
    )
    m = pat["rec"].search(line)
    (
        (
            vals.__setitem__("rec_m", float(m.group(1))),
            vals.__setitem__("rec_s", float(m.group(2))),
        )
        if m
        else None
    )
    m = pat["f1"].search(line)
    (
        (
            vals.__setitem__("f1_m", float(m.group(1))),
            vals.__setitem__("f1_s", float(m.group(2))),
        )
        if m
        else None
    )
    m = pat["saved"].search(line)
    vals["model_path"] = m.group(1) if m else vals["model_path"]

path = pathlib.Path(CSV_PATH)
path.parent.mkdir(parents=True, exist_ok=True)
new_file = not path.exists()

with path.open("a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(
            [
                "timestamp",
                "loaded_rows",
                "downsample_rows",
                "n_features",
                "acc_mean",
                "acc_std",
                "prec_mean",
                "prec_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
                "model_path",
            ]
        )
    w.writerow(
        [
            datetime.datetime.now().isoformat(timespec="seconds"),
            vals["loaded"],
            vals["downsample"],
            vals["n_features"],
            vals["acc_m"],
            vals["acc_s"],
            vals["prec_m"],
            vals["prec_s"],
            vals["rec_m"],
            vals["rec_s"],
            vals["f1_m"],
            vals["f1_s"],
            vals["model_path"],
        ]
    )
