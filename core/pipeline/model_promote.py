from pathlib import Path
import json
import shutil
from core.utils.telegram import (
    send_telegram_message,
    escape_markdown,
)  # уже есть в проекте

ROOT = Path("/root/stockbot")
M = ROOT / "core/models"
L = ROOT / "logs"
L.mkdir(parents=True, exist_ok=True)

new_model = M / "xgb_spike4_v2.json"
new_meta = M / "xgb_spike4_v2.meta.json"
cur_model = M / "xgb_current.json"
cur_meta = M / "xgb_current.meta.json"
logfile = L / "model_registry.log"

if not (new_model.exists() and new_meta.exists()):
    print("[model-promote] nothing to promote (v2 missing)")
    raise SystemExit(0)


def get_ap(meta):  # AP — основной критерий
    return float(meta.get("ap_valid", meta.get("ap", 0.0)))


def get_auc(meta):
    return float(meta.get("auc_valid", meta.get("auc", 0.0)))


new = json.loads(new_meta.read_text())

status = ""
promoted = False
if cur_meta.exists() and cur_model.exists():
    cur = json.loads(cur_meta.read_text())
    if get_ap(new) + 1e-9 >= get_ap(cur):  # better_or_equal
        shutil.copy2(new_model, cur_model)
        shutil.copy2(new_meta, cur_meta)
        status = f"PROMOTED v2 (AP_new={get_ap(new):.4f} >= AP_cur={get_ap(cur):.4f})"
        promoted = True
    else:
        status = f"KEPT current (AP_new={get_ap(new):.4f} < AP_cur={get_ap(cur):.4f})"
else:
    shutil.copy2(new_model, cur_model)
    shutil.copy2(new_meta, cur_meta)
    status = f"PROMOTED (first) AP_new={get_ap(new):.4f}"
    promoted = True

print("[model-registry]", status)
with logfile.open("a") as f:
    f.write(status + "\n")

# Telegram отчёт
try:
    lines = []
    lines.append("🤖 *Elios — Model Promote*")
    lines.append(
        f"📦 `xgb_spike4_v2`  AUC={get_auc(new):.4f} | AP={get_ap(new):.4f} | iters={new.get('best_iteration')}"
    )
    lines.append(
        f"🗓 {escape_markdown(str(new.get('date_min')))} → {escape_markdown(str(new.get('date_max')))} | train={new.get('train_rows')} | valid={new.get('valid_rows')}"
    )
    lines.append("🗂 " + ("*promoted*" if promoted else "*kept current*"))
    send_telegram_message("\n".join(lines))
except Exception as e:
    print(f"[warn] telegram: {e}")
