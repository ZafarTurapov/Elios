# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path

import requests


# --- .env loader (мягкий)
def _load_env(paths=("/root/stockbot/.env", ".env")):
    for fp in paths:
        p = Path(fp)
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'").strip('"')
            if k and not os.environ.get(k):
                os.environ[k] = v


_load_env()


# --- mask helper
def _mask(x, keep=4):
    if not x:
        return "—"
    x = str(x)
    if len(x) <= keep * 2:
        return x[0:2] + "…" + x[-2:]
    return x[:keep] + "…" + x[-keep:]


ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE = os.getenv("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("🔐 Ключи/эндпоинты:")
print(f"  ALPACA_API_KEY     = { _mask(ALPACA_API_KEY) }")
print(f"  ALPACA_SECRET_KEY  = { _mask(ALPACA_SECRET_KEY) }")
print(f"  ALPACA_BASE_URL    = { ALPACA_BASE_URL }")
print(f"  ALPACA_DATA_BASE   = { ALPACA_DATA_BASE }")
print(f"  OPENAI_API_KEY     = { _mask(OPENAI_API_KEY) }")

headers = {
    "APCA-API-KEY-ID": ALPACA_API_KEY or "",
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY or "",
}

ok_alp_acc = ok_alp_data = ok_oai = False
err_alp_acc = err_alp_data = err_oai = None

# --- Alpaca /v2/account
try:
    r = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=headers, timeout=10)
    ok_alp_acc = r.status_code == 200 and isinstance(r.json(), dict)
    if not ok_alp_acc:
        err_alp_acc = f"HTTP {r.status_code} {r.text[:120]}"
except Exception as e:
    err_alp_acc = str(e)

# --- Alpaca quotes latest (AAPL)
try:
    r = requests.get(
        f"{ALPACA_DATA_BASE}/stocks/AAPL/quotes/latest",
        params={"feed": "iex"},
        headers=headers,
        timeout=10,
    )
    ok_alp_data = r.status_code == 200 and "quote" in (r.json() or {})
    if not ok_alp_data:
        err_alp_data = f"HTTP {r.status_code} {r.text[:120]}"
except Exception as e:
    err_alp_data = str(e)

# --- OpenAI tiny test (chat.completions)
try:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    out = client.chat.completions.create(
        model=os.getenv("ELIOS_SIGNAL_GPT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        temperature=0.0,
    )
    ok_oai = bool(out and out.choices)
except Exception as e:
    err_oai = str(e)[:200]

print(
    "\n✅ Alpaca account OK" if ok_alp_acc else f"❌ Alpaca account FAIL: {err_alp_acc}"
)
print("✅ Alpaca data OK" if ok_alp_data else f"❌ Alpaca data FAIL: {err_alp_data}")
print("✅ OpenAI OK" if ok_oai else f"❌ OpenAI FAIL: {err_oai}")

# --- запуск signal_engine с ФИЛЬТРАМИ (NO_FILTERS=0). GPT включён, чтобы проверить ключи в реальном ходе
env = os.environ.copy()
env["PYTHONPATH"] = "/root/stockbot"
env["ELIOS_FORCE_OPEN"] = "1"  # чтобы не зависеть от календаря
env["ELIOS_NO_FILTERS"] = "0"  # Фильтры включены
env["ELIOS_NO_GPT"] = "0"  # Проверяем GPT в конвейере
env["ELIOS_DEBUG"] = env.get("ELIOS_DEBUG", "0")

print("\n🚀 Старт signal_engine.py с фильтрами (ELIOS_NO_FILTERS=0, ELIOS_NO_GPT=0)…")
import subprocess

ret = subprocess.run([sys.executable, "-u", "core/trading/signal_engine.py"], env=env)
print(f"signal_engine.py exit={ret.returncode}")

# --- Нормализуем signals.json в новый стандарт (на случай legacy-вывода)
sigp = Path("core/trading/signals.json")
if sigp.exists():
    try:
        obj = json.loads(sigp.read_text(encoding="utf-8"))
    except Exception:
        obj = {}
    if isinstance(obj, dict) and isinstance(obj.get("signals"), list):
        print(f"ℹ️ signals.json уже стандартный (signals={len(obj['signals'])})")
    else:
        try:
            from core.trading.signals.io import normalize_signals, save_signals

            if isinstance(obj, dict):
                seq = [{"symbol": k, **(v or {})} for k, v in obj.items()]
            elif isinstance(obj, list):
                seq = obj
            else:
                seq = []
            payload = normalize_signals(seq)
            save_signals(payload, path=str(sigp))
            print(
                f"✅ Нормализовано → {sigp} (signals={len(payload.get('signals',[]))})"
            )
        except Exception as e:
            print(f"❗ Не удалось нормализовать signals.json: {e}")
else:
    print("❗ signals.json не найден после прогона")

# --- Короткая сводка
try:
    obj = json.loads(Path("core/trading/signals.json").read_text(encoding="utf-8"))
    cnt = len(obj.get("signals", [])) if isinstance(obj, dict) else 0
    print(f"\n📦 Итог: signals={cnt}")
    if cnt:
        first = obj["signals"][0]
        print(
            "▶ sample:",
            {k: first.get(k) for k in ("symbol", "price", "action", "weight")},
        )
except Exception as e:
    print(f"❗ Ошибка чтения итогового файла: {e}")
