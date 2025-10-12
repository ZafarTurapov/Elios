# -*- coding: utf-8 -*-
import os, sys, json, re, py_compile, subprocess, socket
from pathlib import Path
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import List

ROOT = Path("/root/stockbot")
TZ   = ZoneInfo("Asia/Tashkent")

FILES = [
    "core/trading/signal_engine.py",
    "core/trading/trade_executor.py",
    "core/trading/positions_sync.py",
    "core/trading/sell_notifier.py",
    "core/trading/eod_flatten.py",
    "core/trading/loop_launcher.py",
    "core/trading/body_packer.py",
    "core/utils/telegram.py",
    "core/utils/market_calendar.py",
]

ENV_KEYS = ["ALPACA_API_KEY","ALPACA_SECRET_KEY","OPENAI_API_KEY"]

def info(msg):  print(f"ℹ️  {msg}")
def ok(msg):    print(f"✅ {msg}")
def warn(msg):  print(f"⚠️  {msg}")
def bad(msg):   print(f"❌ {msg}")

def sh(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def load_dotenv_into_env(dotenv: Path):
    if not dotenv.exists(): return
    for line in dotenv.read_text(encoding="utf-8", errors="ignore").splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        m=re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if not m: continue
        k,v=m.group(1), m.group(2)
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v=v[1:-1]
        os.environ.setdefault(k, v)

def compile_files():
    issues=0
    for rel in FILES:
        p=(ROOT/rel)
        if not p.exists():
            warn(f"Файл не найден: {rel}")
            issues+=1
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            ok(f"Синтаксис ок: {rel}")
        except Exception as e:
            bad(f"Синтакс-ошибка в {rel}: {e}")
            issues+=1
    return issues

def check_env():
    # подгружаем .env.local, если есть
    load_dotenv_into_env(ROOT/".env.local")
    miss=[k for k in ENV_KEYS if not os.environ.get(k)]
    if miss:
        warn("Отсутствуют переменные окружения: " + ", ".join(miss))
        return len(miss)
    ok("Ключи окружения найдены (Alpaca/OpenAI).")
    return 0

def internet_ok(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

def check_alpaca():
    import urllib.request, json as _json
    key=os.environ.get("ALPACA_API_KEY")
    sec=os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        warn("Пропуск проверки Alpaca (нет ключей).")
        return 1
    base=os.environ.get("ALPACA_API_BASE_URL","https://paper-api.alpaca.markets")
    url=f"{base.rstrip('/')}/v2/account"
    req=urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": sec,
        "User-Agent": "EliosPreflight/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data=_json.loads(r.read().decode("utf-8"))
            if "status" in data:
                ok(f"Alpaca OK: статус аккаунта = {data.get('status')}")
                return 0
            else:
                warn("Alpaca ответ без поля status — проверь ключи/URL.")
                return 1
    except Exception as e:
        bad(f"Alpaca не отвечает: {e}")
        return 1

def check_systemd():
    issues=0
    for unit in ["elios-loop.timer","sell-notifier.timer","eod-flatten.timer"]:
        r=sh(["systemctl","is-enabled",unit])
        if r.returncode==0 and r.stdout.strip() in ("enabled","static"):
            ok(f"unit включен: {unit}")
        else:
            warn(f"unit не включен: {unit} -> {r.stdout.strip()}")
            issues+=1
    # показать ближайшие срабатывания
    r=sh(["systemctl","list-timers","--all","--no-pager"])
    print(r.stdout)
    return issues

def market_calendar_probe():
    try:
        sys.path.append(str(ROOT))
        from core.utils.market_calendar import is_market_open_today
        opened = is_market_open_today()
        ok(f"Рынок сегодня {'ОТКРЫТ' if opened else 'закрыт'} по календарю.")
        return 0
    except Exception as e:
        warn(f"Календарь рынка недоступен/упал: {e}")
        return 1

def main():
    print("=== Elios Preflight ===")
    now=datetime.now(TZ)
    print(f"Время: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    # 1) Синтаксис
    issues=compile_files()
    # 2) Окружение
    issues+=check_env()
    # 3) Интернет
    if internet_ok():
        ok("Интернет: OK")
        # 4) Alpaca
        issues+=check_alpaca()
    else:
        warn("Нет интернета — пропуск Alpaca проверки.")
        issues+=1
    # 5) systemd
    issues+=check_systemd()
    # 6) Календарь рынка
    issues+=market_calendar_probe()

    if issues==0:
        ok("PRE-FLIGHT: ГОТОВО К ЗАПУСКУ ✅")
        sys.exit(0)
    else:
        bad(f"PRE-FLIGHT: найдено проблем: {issues} — см. вывод выше.")
        sys.exit(1)

if __name__=="__main__":
    main()
