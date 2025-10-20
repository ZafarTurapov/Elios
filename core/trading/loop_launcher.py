import os
import subprocess
import time
from datetime import datetime
from datetime import time as dtime
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# .env bootstrap
from dotenv import find_dotenv, load_dotenv

from core.utils import (
    env_keys as _envkeys,
)  # noqa: F401  (подтягивает env-ключи в рантайме)
from core.utils.alpaca_headers import alpaca_headers

# === прогреть ключи/утилиты ===
from core.utils.market_calendar import is_market_open_today
from core.utils.telegram import send_telegram_message

# ----------------- ENV / .env -----------------
try:
    # сначала базовый .env, потом найденный в текущем дереве (если есть)
    load_dotenv("/root/stockbot/.env")
    env_file = find_dotenv(usecwd=True)
    if env_file:
        load_dotenv(env_file, override=True)
except Exception:
    pass

# Безопасные дефолты Alpaca endpoints (если не заданы)
os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
os.environ.setdefault("ALPACA_DATA_BASE", "https://data.alpaca.markets/v2")
os.environ.setdefault("ALPACA_NEWS_BASE", "https://data.alpaca.markets/v1beta1")

# ----------------- Константы рынка (Asia/Tashkent) -----------------
# Торги: Пн–Пт 18:30–01:00
MARKET_OPEN = dtime(18, 30)
MARKET_CLOSE = dtime(1, 0)  # хвост до 01:00
EOD_BUFFER_MIN = 5  # окно EOD-продаж за 5 минут до MARKET_CLOSE

# BUY-окно внутри торговых часов
BUY_WIN_START_STR = os.getenv("BUY_WINDOW_START", "18:30")  # локальное время Ташкент
BUY_WIN_END_STR = os.getenv("BUY_WINDOW_END", "21:00")

# Каденсы
SELL_EVERY_MIN = int(os.getenv("SELL_EVERY_MIN", "5"))  # SELL из лупа (обычно выключен)
BUY_EVERY_MIN = int(os.getenv("BUY_EVERY_MIN", "10"))  # сигнал/экзекутор

# Флаги SELL из лупа (по умолчанию SELL запускается таймером вне лупа)
LOOP_SELL_ENABLED = os.getenv("LOOP_SELL_ENABLED", "0") == "1"
LOOP_EOD_SELL_ENABLED = os.getenv("LOOP_EOD_SELL_ENABLED", "0") == "1"

# ----------------- Служебные переменные цикла -----------------
eod_done_ordinal = -1
last_slot_buy_id = -1
last_slot_sell_id = -1
last_force_sell_date = None
notified_off_hours = False

# ----------------- Пути -----------------
PYTHON = "/root/stockbot/venv/bin/python"
ROOT = Path("/root/stockbot")
LOG_DIR = ROOT / "logs"
LOCK_DIR = LOG_DIR / "locks"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOCK_DIR.mkdir(parents=True, exist_ok=True)

# DEBUG: проверка наличия ключей в окружении (значение не печатаем)
api_key_check = (
    os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or ""
).strip()
print(f"[DEBUG] ENV KEY: {'✅ Найден' if api_key_check else '❌ Не найден'}")


# ----------------- helpers -----------------
def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def within_buy_window(now_tz: datetime) -> bool:
    sh, sm = _parse_hhmm(BUY_WIN_START_STR)
    eh, em = _parse_hhmm(BUY_WIN_END_STR)
    start = now_tz.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now_tz.replace(hour=eh, minute=em, second=0, microsecond=0)
    return start <= now_tz < end


def _slot_id(now_tz: datetime, step_min: int) -> int:
    """Уникальный ID N-минутного слота в рамках суток."""
    day_ord = now_tz.date().toordinal()
    minute_of_day = now_tz.hour * 60 + now_tz.minute
    slot = minute_of_day // max(1, step_min)
    return day_ord * (1440 // max(1, step_min)) + slot


def is_market_hours() -> bool:
    """Проверка попадания 'сейчас' в торговое окно, учитывая перелом суток."""
    now = datetime.now(ZoneInfo("Asia/Tashkent")).time()
    if MARKET_OPEN <= MARKET_CLOSE:
        # обычное окно внутри суток
        return MARKET_OPEN <= now <= MARKET_CLOSE
    else:
        # окно через полночь (наш случай 18:30–01:00)
        return now >= MARKET_OPEN or now <= MARKET_CLOSE


def in_eod_window(now_tz: datetime) -> bool:
    """True, если мы в EOD-окне за EOD_BUFFER_MIN минут до MARKET_CLOSE."""
    # MARKET_CLOSE у нас 01:00 локально — дата *сегодня*.
    # За 5 минут до 01:00 => 00:55..01:00
    sell_start_dt = datetime.combine(now_tz.date(), MARKET_CLOSE) - timedelta(
        minutes=EOD_BUFFER_MIN
    )
    end_dt = datetime.combine(now_tz.date(), MARKET_CLOSE)
    return sell_start_dt.time() <= now_tz.time() < end_dt.time()


def _lock_path(name: str) -> Path:
    return LOCK_DIR / f"{name}.pid"


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _can_start_task(lock_name: str) -> bool:
    """Не даём запустить два одинаковых фоновых задания (по PID-локу)."""
    lp = _lock_path(lock_name)
    if not lp.exists():
        return True
    try:
        txt = lp.read_text().strip()
        pid = int(txt.split()[0])
    except Exception:
        lp.unlink(missing_ok=True)
        return True
    if _is_pid_alive(pid):
        return False
    lp.unlink(missing_ok=True)
    return True


def _set_lock(lock_name: str, pid: int):
    try:
        _lock_path(lock_name).write_text(f"{pid} {int(time.time())}\n")
    except Exception:
        pass


def run_module(name: str, module_path: str):
    """Блокирующий запуск (дождаться завершения).
    module_path — строка вида '-u -m core.trading.signal_engine'
    """
    print(f"[{datetime.now()}] ▶️ Запуск: {name}")
    try:
        send_telegram_message(f"⚙️ Запущен модуль: {name}")
        # Разделим строку по пробелам и передадим аргументы python
        subprocess.run(
            [PYTHON] + module_path.split(), env=os.environ.copy(), check=False
        )
    except Exception as e:
        print(f"❌ Ошибка запуска {name}: {e}")
        send_telegram_message(f"⛔️ Ошибка запуска {name}: {e}")


def run_module_bg(name: str, module_path: str, lock_name: str | None = None):
    """Неблокирующий запуск (в фоне) с PID-локом."""
    if lock_name and not _can_start_task(lock_name):
        print(f"[{datetime.now()}] ⏩ {name} пропущен — уже выполняется")
        return
    print(f"[{datetime.now()}] ▶️ Фоновый запуск: {name}")
    try:
        send_telegram_message(f"⚙️ (BG) Запущен модуль: {name}")
        proc = subprocess.Popen(
            [PYTHON] + module_path.split(),
            env=os.environ.copy(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        if lock_name:
            _set_lock(lock_name, proc.pid)
    except Exception as e:
        print(f"❌ Ошибка фонового запуска {name}: {e}")
        send_telegram_message(f"⛔️ Ошибка фонового запуска {name}: {e}")


def _fallback_flatten_all():
    """Резервная принудительная ликвидация (на случай фейла sell_engine)."""
    import requests

    try:
        key = (
            os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or ""
        ).strip()
        sec = (
            os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or ""
        ).strip()
        base = (
            os.getenv("ALPACA_API_BASE_URL")
            or os.getenv("ALPACA_BASE_URL")
            or "https://paper-api.alpaca.markets"
        ).strip()
        if not key or not sec:
            raise RuntimeError("ALPACA keys not set")
        h = alpaca_headers()
        r = requests.delete(
            f"{base}/v2/positions",
            params={"cancel_orders": "true"},
            headers=h,
            timeout=15,
        )
        print(f"[EOD] fallback flatten: {r.status_code}")
        send_telegram_message(f"💣 Fallback flatten: {r.status_code}")
    except Exception as e:
        print(f"[EOD] fallback error: {e}")
        send_telegram_message(f"⛔️ Fallback flatten error: {e}")


# ----------------- main loop -----------------
def main_loop():
    # single-instance guard
    if not _can_start_task("loop"):
        print("[LOCK] loop already running — exit")
        return
    _set_lock("loop", os.getpid())

    global notified_off_hours, last_slot_buy_id, last_slot_sell_id, eod_done_ordinal, last_force_sell_date

    print(
        "🌀 Запуск торгового цикла Искры (SELL — внешний таймер; здесь: Signal/Executor)"
    )
    send_telegram_message(
        "🚀 Торговый цикл Искры запущен (SELL — внешний таймер; здесь: Signal/Executor)"
    )

    if not is_market_open_today():
        msg = "⛔️ Сегодня рынок закрыт. Искра завершает цикл."
        print(msg)
        send_telegram_message(msg)
        return

    print("🔁 Вход в цикл while True")
    while True:
        now = datetime.now(ZoneInfo("Asia/Tashkent"))
        minute = now.minute
        hour = now.hour

        # === EOD окно: по флагу (за EOD_BUFFER_MIN минут до закрытия)
        if (
            LOOP_EOD_SELL_ENABLED
            and is_market_hours()
            and in_eod_window(now)
            and eod_done_ordinal != now.date().toordinal()
        ):
            print(f"[{datetime.now()}] ▶️ Запуск: EOD SELL (буфер {EOD_BUFFER_MIN}m)")
            send_telegram_message(
                f"💼 EOD SELL: завершаем позиции за {EOD_BUFFER_MIN} мин до закрытия"
            )
            try:
                run_module("EOD SELL", "-u -m core.trading.sell_engine")
            except Exception as e:
                print(f"[WARN] sell_engine failed: {e}")
                _fallback_flatten_all()
            eod_done_ordinal = now.date().toordinal()

        slot_buy_id = _slot_id(now, BUY_EVERY_MIN)
        slot_sell_id = _slot_id(now, SELL_EVERY_MIN)
        print(
            f"⏱️ {now.strftime('%H:%M:%S')} (buy_slot={slot_buy_id}, sell_slot={slot_sell_id})"
        )

        # Вне торговых часов — спим и один раз уведомляем
        if not is_market_hours():
            if not notified_off_hours:
                msg = f"🕔 Вне торговых часов: {now.time()}. Искра спит."
                print(msg)
                send_telegram_message(msg)
                notified_off_hours = True
            time.sleep(30)
            continue
        else:
            notified_off_hours = False

        # === SELL из лупа — только если явно включён флагом и не в EOD-окне
        if (
            LOOP_SELL_ENABLED
            and (minute % SELL_EVERY_MIN == 0)
            and (last_slot_sell_id != slot_sell_id)
            and not in_eod_window(now)
        ):
            run_module_bg(
                "SELL ENGINE", "-u -m core.trading.sell_engine", lock_name="sell"
            )
            last_slot_sell_id = slot_sell_id

        # === Каждые BUY_EVERY_MIN: сигналы → исполнение → синхронизация (ТОЛЬКО в BUY-окне)
        if (
            minute % BUY_EVERY_MIN == 0
            and last_slot_buy_id != slot_buy_id
            and not (
                hour == MARKET_CLOSE.hour and minute == MARKET_CLOSE.minute
            )  # не в момент выхода
            and not in_eod_window(now)
            and within_buy_window(now)
        ):
            run_module("SIGNAL ENGINE", "-u -m core.trading.signal_engine")
            run_module("TRADE EXECUTOR", "-u -m core.trading.trade_executor")
            run_module(
                "POSITIONS SYNC (после EXECUTOR)", "-u -m core.trading.positions_sync"
            )
            last_slot_buy_id = slot_buy_id

        # Завершение торгового окна (в 01:00)
        if hour == MARKET_CLOSE.hour and minute == MARKET_CLOSE.minute:
            send_telegram_message("📉 Торги завершены. Искра засыпает.")
            break

        # === EOD FORCE SELL за 5 минут до закрытия — тоже по флагу
        if (
            LOOP_EOD_SELL_ENABLED
            and (hour == MARKET_CLOSE.hour and minute == (MARKET_CLOSE.minute - 5))
            and (last_force_sell_date != now.date())
        ):
            print(f"[{datetime.now()}] ▶️ Запуск: SELL ENGINE (EOD 00:55)")
            send_telegram_message("💣 FORCE SELL (за 5 минут до закрытия)")
            run_module("SELL ENGINE", "-u -m core.trading.sell_engine")
            run_module(
                "Финальная синхронизация (после SELL)",
                "-u -m core.trading.positions_sync",
            )
            last_force_sell_date = now.date()

        time.sleep(2)


def _print_env_masked():
    def _mask(k: str) -> str:
        v = os.getenv(k, "")
        return (v[:4] + "…" + v[-4:]) if len(v) >= 12 else ("*" * len(v))

    print("ALPACA_BASE_URL =", os.getenv("ALPACA_BASE_URL"))
    print("ALPACA_API_KEY  =", _mask("ALPACA_API_KEY"))
    print("APCA_API_KEY_ID =", _mask("APCA_API_KEY_ID"))
    print("ALPACA_SECRET   =", _mask("ALPACA_SECRET_KEY"))
    print("APCA_API_SECRET =", _mask("APCA_API_SECRET_KEY"))
    print("ALPACA_DATA_BASE=", os.getenv("ALPACA_DATA_BASE"))
    print("ALPACA_NEWS_BASE=", os.getenv("ALPACA_NEWS_BASE"))


if __name__ == "__main__":
    import sys

    if "--print-env" in sys.argv:
        _print_env_masked()
        sys.exit(0)

    try:
        main_loop()
    except Exception as e:
        send_telegram_message(f"⛔️ Искра аварийно завершила цикл: {e}")
        raise
