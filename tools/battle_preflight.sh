#!/usr/bin/env bash
set -euo pipefail
cd /root/stockbot
PY="/root/stockbot/venv/bin/python"
export PYTHONPATH=/root/stockbot

# Подтягиваем переменные как делает systemd EnvironmentFile
if [ -f ./.env.local ]; then
  set -a
  . ./.env.local
  set +a
fi

echo "===== 1) Versions ====="
$PY - <<'PY'
import platform
def ver(p):
    try:
        import importlib.metadata as m; print(p, m.version(p))
    except Exception:
        print(p, "— n/a")
print("Python:", platform.python_version())
for p in ["openai","requests","pandas","ta"]:
    ver(p)
PY
echo

echo "===== 2) Alpaca /v2/account ping ====="
$PY - <<'PY'
import os, requests
base=os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets")
k=os.getenv("ALPACA_API_KEY",""); s=os.getenv("ALPACA_SECRET_KEY","")
def mask(x): 
    return (x[:4]+"…"+x[-3:]) if len(x)>=8 else ("EMPTY" if not x else "SHORT")
print("BASE:", base)
print("KEYS:", mask(k), mask(s))
hdr={"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s,"Accept":"application/json"}
try:
    r=requests.get(f"{base}/v2/account",headers=hdr,timeout=10)
    print("HTTP", r.status_code)
    ct=r.headers.get("content-type","")
    j=r.json() if "application/json" in ct else {}
    print("status:", j.get("status"), "| currency:", j.get("currency"), "| trading_blocked:", j.get("trading_blocked"))
    print("equity:", j.get("equity"), "| cash:", j.get("cash"), "| buying_power:", j.get("buying_power"))
    if r.status_code!=200:
        print("Body:", j or r.text[:300])
except Exception as e:
    print("ERROR:", e)
PY
echo

echo "===== 3) GPT quick ping (sell_engine.gpt_confirm_sell) ====="
$PY - <<'PY'
from core.trading.sell_engine import gpt_confirm_sell
ok = gpt_confirm_sell("AAPL", 100.0, 101.0, (101.0-100.0)/100.0, 1.0)
print("gpt_confirm_sell returned:", ok)
PY
echo

echo "===== 4) Telegram ping ====="
$PY - <<'PY'
from core.utils.telegram import send_telegram_message
send_telegram_message("🧪 Elios preflight: Telegram OK")
print("sent")
PY
echo

echo "===== 5) positions_sync run ====="
$PY -u -m core.trading.positions_sync && echo "positions_sync: ✅ OK" || echo "positions_sync: ❌ FAIL"
echo

echo "===== 6) Timers snapshot ====="
systemctl list-timers --all | egrep -i 'sell-health-probe|elios-account-sync|elios-loop' || true
echo
echo "===== Preflight done ====="
