# -*- coding: utf-8 -*-
import io
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

CACHE_DIR = Path("logs/cache/squeeze")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TTL_SEC_DEFAULT = int(os.getenv("ELIOS_SQ_TTL_SEC", "21600"))  # 6h
FINRA_ENABLED = os.getenv("ELIOS_SQ_FINRA_ENABLED", "1") == "1"


# ---------- NASDAQ Short Interest (free) ----------
def _nasdaq_short_interest(symbol: str, timeout=8):
    """
    Пытаемся получить:
      - si_pct  (Short Interest % of Float)
      - dtc     (Days to Cover)
    Сначала официальный JSON, затем HTML-страница NASDAQ.
    Возвращает dict {'si_pct': float|0.0, 'dtc': float|0.0} либо None.
    """
    import re

    import requests

    sym = str(symbol).upper().strip()

    def _as_float(x):
        try:
            if isinstance(x, str):
                x = x.replace(",", "").replace("%", "").strip()
            return float(x)
        except Exception:
            return None

    # JSON API (часто требует UA и может 403)
    try:
        url = f"https://api.nasdaq.com/api/quote/{sym}/short-interest?assetclass=stocks"
        hdr = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": f"https://www.nasdaq.com/market-activity/stocks/{sym}/short-interest",
            "Origin": "https://www.nasdaq.com",
        }
        r = requests.get(url, headers=hdr, timeout=timeout)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith(
            "application/json"
        ):
            j = r.json() or {}
            data = j.get("data") or {}
            # Поля в этом JSON отличаются от времени к времени, пробуем вытащить ratio и/или percent
            # Пробуем Days to Cover
            dtc = None
            si_pct = None
            # Иногда лежит в data['rows'] как список словарей
            rows = data.get("rows") or data.get("shortInterest") or []
            if isinstance(rows, list):
                for row in rows:
                    name = str(row.get("name") or row.get("label") or "").lower()
                    val = (
                        row.get("value")
                        or row.get("v")
                        or row.get("raw")
                        or row.get("text")
                    )
                    if "days to cover" in name:
                        dtc = _as_float(val)
                    if "% of float" in name or "short interest %" in name:
                        si_pct = _as_float(val)
            # Иногда значения лежат в data['shortInterest'] как dict
            if si_pct is None:
                v = data.get("percentOfFloat") or data.get("shortPercentOfFloat")
                si_pct = _as_float(v)
            if dtc is None:
                v = data.get("daysToCover") or data.get("shortInterestRatio")
                dtc = _as_float(v)
            if (si_pct is not None) or (dtc is not None):
                return {"si_pct": float(si_pct or 0.0), "dtc": float(dtc or 0.0)}
    except Exception:
        pass

    # HTML fallback: парсим страницу /short-interest
    try:
        url = f"https://www.nasdaq.com/market-activity/stocks/{sym}/short-interest"
        hdr = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=hdr, timeout=timeout)
        if r.status_code == 200 and r.text:
            html = r.text
            # Ищем куски вроде "Short Interest Ratio (Days To Cover)  1.23"
            m_dtc = re.search(
                r"Short\s+Interest\s+Ratio.*?([\d.,]+)", html, re.I | re.S
            )
            m_pct = re.search(
                r"(Short\s+Interest\s*%|% of Float).*?([\d.,]+)\s*%", html, re.I | re.S
            )
            dtc = _as_float(m_dtc.group(1)) if m_dtc else None
            si_pct = _as_float(
                m_pct.group(2)
                if m_pct and m_pct.lastindex >= 2
                else (m_pct.group(1) if m_pct else None)
            )
            if (si_pct is not None) or (dtc is not None):
                return {"si_pct": float(si_pct or 0.0), "dtc": float(dtc or 0.0)}
    except Exception:
        pass

    return None


# Простая файловая прослойка
def _cache_path(sym: str) -> Path:
    return CACHE_DIR / f"{sym.upper()}.json"


def _load_cache(sym: str, ttl=TTL_SEC_DEFAULT):
    p = _cache_path(sym)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        if (time.time() - float(obj.get("_ts", 0))) <= ttl:
            return obj.get("data")
    except Exception:
        return None
    return None


def _save_cache(sym: str, data: dict):
    try:
        _cache_path(sym).write_text(
            json.dumps({"_ts": time.time(), "data": data}, ensure_ascii=False)
        )
    except Exception:
        pass


# ---------- yfinance float / ADV helpers ----------
def _yf_hist(symbol: str, period="45d", interval="1d"):
    import yfinance as yf

    df = yf.download(
        symbol, period=period, interval=interval, progress=False, auto_adjust=False
    )
    return df


def _float_shares(symbol: str) -> float:
    """
    Пытаемся получить свободный float (шт.):
    1) Ticker.fast_info: shares_float / float_shares / free_float / shares_outstanding(+held%)
    2) Ticker.get_info(): floatShares / sharesFloat / impliedSharesOutstanding / sharesOutstanding (+held%)
    3) Ticker.get_shares_full(): берём последнюю sharesOutstanding и уменьшаем на held%
    4) Последний fallback: 0.7 * shares_outstanding (очень консервативно)
    """
    try:
        import yfinance as yf

        t = yf.Ticker(symbol)
        # --- fast_info сначала
        try:
            fi = getattr(t, "fast_info", None)
            fi_d = fi if isinstance(fi, dict) else (dict(fi.__dict__) if fi else {})
            for k in ("shares_float", "float_shares", "free_float"):
                fs = (fi_d or {}).get(k)
                if fs and float(fs) > 0:
                    return float(fs)
            so = (fi_d or {}).get("shares_outstanding") or (fi_d or {}).get(
                "implied_shares_outstanding"
            )
        except Exception:
            fi_d, so = {}, None

        # --- get_info fallback
        held_ins = None
        held_inst = None
        try:
            info = t.get_info(retries=1)
            # прямой float
            for k in ("floatShares", "sharesFloat", "freeFloat"):
                fs = info.get(k)
                if fs and float(fs) > 0:
                    return float(fs)
            # запоминаем возможные проценты владения
            held_ins = info.get("heldPercentInsiders")
            held_inst = info.get("heldPercentInstitutions")
            # возможные outstanding
            so = (
                so
                or info.get("sharesOutstanding")
                or info.get("impliedSharesOutstanding")
            )
        except Exception:
            info = {}

        # --- time series: shares outstanding (последняя точка)
        if not so:
            try:
                sh = t.get_shares_full(start=None)
                if (
                    sh is not None
                    and getattr(sh, "empty", False) is False
                    and len(sh) > 0
                ):
                    last = sh.dropna().iloc[-1]
                    # pandas Series -> scalar
                    try:
                        so = float(getattr(last, "item", lambda: float(last))())
                    except Exception:
                        so = float(last)
            except Exception:
                pass

        # --- если есть shares_outstanding, оценим флоат по held%
        if so and float(so) > 0:
            so = float(so)
            # если есть проценты владения — снимем их; иначе оставим консервативно 70% от SO
            frac_free = None
            try:
                parts = [
                    x
                    for x in (held_ins, held_inst)
                    if isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1
                ]
                if parts:
                    frac_free = max(
                        0.0, 1.0 - min(1.0, sum(parts))
                    )  # грубо: инсайдеры+институты
            except Exception:
                frac_free = None
            if frac_free is None or frac_free == 0:
                frac_free = 0.7  # очень консервативно
            fs_est = so * frac_free
            if fs_est > 0:
                return float(fs_est)

    except Exception:
        pass
    return 0.0


def _adv30(symbol: str) -> float:
    """Средний дневной объём за ~30 дней (акции)."""
    try:
        df = _yf_hist(symbol, period="70d", interval="1d")
        if df is None or df.empty:
            return 0.0
        v = df["Volume"].dropna().tail(30)
        if len(v) == 0:
            return 0.0
        return float(v.mean())
    except Exception:
        return 0.0


def _today_ohlc(symbol: str):
    """Возвращаем (prev_close, today_low, today_close, today_volume)."""
    try:
        df = _yf_hist(symbol, period="5d", interval="1d")
        if df is None or df.empty or len(df) < 2:
            return (None, None, None, None)
        prev = df["Close"].iloc[-2]
        # prev может быть scalar или 1-элементная Series (редкий кейс у yfinance); аккуратно:
        try:
            prev_close = float(prev.item())  # numpy scalar
        except Exception:
            try:
                prev_close = float(prev.iloc[0])  # 1-элементная Series
            except Exception:
                prev_close = float(prev)

        row = df.iloc[-1]
        low_v = row["Low"]
        close_v = row["Close"]
        vol_v = row["Volume"]
        # Точно в float:
        try:
            low = float(getattr(low_v, "item", lambda: low_v)())
        except Exception:
            low = float(low_v)
        try:
            close = float(getattr(close_v, "item", lambda: close_v)())
        except Exception:
            close = float(close_v)
        try:
            vol = float(getattr(vol_v, "item", lambda: vol_v)())
        except Exception:
            vol = float(vol_v)
        return (prev_close, low, close, vol)
    except Exception:
        return (None, None, None, None)
        prev_close = float(df["Close"].iloc[-2])
        today = df.iloc[-1]
        low = float(today["Low"])
        close = float(today["Close"])
        vol = float(today["Volume"])
        return (prev_close, low, close, vol)
    except Exception:
        return (None, None, None, None)


# ---------- FINRA Reg SHO daily short volume ----------
# Ежедневные файлы по TRF-потокам. Мы суммируем по нескольким источникам, пока не соберём данные.
_FINRA_SOURCES = (
    "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt",  # Nasdaq TRF (Carteret)
    "https://cdn.finra.org/equity/regsho/daily/FNSQshvol{date}.txt",  # Nasdaq TRF (Chicago)
    "https://cdn.finra.org/equity/regsho/daily/FNRAshvol{date}.txt",  # NYSE TRF
)


def _finra_fetch_for_date(sym: str, yyyymmdd: str):
    short_vol = 0
    total_vol = 0
    for url_t in _FINRA_SOURCES:
        url = url_t.format(date=yyyymmdd)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            # Формат строк: Symbol|ShortVolume|ShortExemptVolume|TotalVolume|...
            for line in io.StringIO(r.text):
                parts = line.strip().split("|")
                if len(parts) < 4:
                    continue
                if parts[0].upper() != sym.upper():
                    continue
                try:
                    sv = int(parts[1])
                    tv = int(parts[3])
                    short_vol += sv
                    total_vol += tv
                except Exception:
                    continue
        except Exception:
            continue
    return short_vol, total_vol


def _finra_short_ratio(sym: str):
    """Доля коротких сделок за вчера (не SI!). Ищем D-1, D-2, ... до 5 дней назад."""
    if not FINRA_ENABLED:
        return None
    # Берём «вчера» относительно Нью-Йорка (чтобы не упереться в ещё не выложенный файл)
    now_utc = datetime.now(timezone.utc)
    for d in range(1, 6):
        day = (now_utc - timedelta(days=d)).astimezone(timezone.utc)
        yyyymmdd = day.strftime("%Y%m%d")
        sv, tv = _finra_fetch_for_date(sym, yyyymmdd)
        if tv > 0:
            return float(sv) / float(tv)
    return None


# ---------- Главная функция ----------


def _finviz_short(symbol: str):
    """Парсит finviz Short Float (%) и Short Ratio (Days to Cover)."""
    try:
        import re as _re

        import requests as _rq

        url = f"https://finviz.com/quote.ashx?t={symbol}"
        r = _rq.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return None, None
        html = r.text
        m_si = _re.search(r"Short Float</td><td.*?>([\d\.]+)%", html, _re.I | _re.S)
        m_dtc = _re.search(r"Short Ratio</td><td.*?>([\d\.]+)\s*<", html, _re.I | _re.S)
        si = float(m_si.group(1)) if m_si else None
        dtc = float(m_dtc.group(1)) if m_dtc else None
        if si is not None and (si < 0 or si > 100):
            si = None
        if dtc is not None and (dtc < 0 or dtc > 100):
            dtc = None
        return si, dtc
    except Exception:
        return None, None


def _iborrowdesk_fee(symbol: str):
    """Берёт оценку borrow fee с iBorrowDesk (если есть)."""
    try:
        import requests as _rq

        r = _rq.get(f"https://iborrowdesk.com/api/ticker/{symbol}", timeout=10)
        if r.status_code != 200:
            return None
        j = r.json() or {}
        fee = j.get("fee")
        try:
            fee = float(fee)
        except Exception:
            fee = None
        if fee is not None and (fee < 0 or fee > 200):
            fee = None
        return fee  # %
    except Exception:
        return None


def get_squeeze_features(
    symbol: str,
    day_volume: float = 0.0,
    allow_stale_cache: bool = True,
    use_nasdaq: bool = False,
    **kwargs,
):
    """
    Лёгкая бесплатная версия squeeze-метрик:
      • float (шт) — yfinance.fast_info / get_info
      • ADV (среднедневной объём) — yfinance history
      • sharesShort — yfinance.get_info (если есть)
      • SI% = sharesShort / float * 100
      • DTC = sharesShort / ADV
      • Util% ~ day_volume / float * 100  (оценка дневной "ротации" флоата)
      • Fee% — fallback с iBorrowDesk (если хелпер доступен и включён)
      • Finviz — fallback для SI/DTC (если хелпер доступен и включён)
    Всё с локальным кешем; при ошибках — мягкая деградация к 0.
    """
    import os

    sym = str(symbol or "").upper().strip()
    if not sym:
        return {
            "squeeze_score": 0.0,
            "si_pct": 0.0,
            "dtc": 0.0,
            "float_m": 0.0,
            "fee_pct": 0.0,
            "util_pct": 0.0,
            "ssr_flag": False,
            "is_squeeze_long_risk": False,
            "is_squeeze_short_opportunity": False,
        }

    # --- cache load ---
    data_cached = None
    try:
        if allow_stale_cache:
            data_cached = _load_cache(sym)
    except Exception:
        data_cached = None
    if isinstance(data_cached, dict):
        return data_cached

    out = {
        "squeeze_score": 0.0,
        "si_pct": 0.0,
        "dtc": 0.0,
        "float_m": 0.0,
        "fee_pct": 0.0,
        "util_pct": 0.0,
        "ssr_flag": False,
        "is_squeeze_long_risk": False,
        "is_squeeze_short_opportunity": False,
    }

    # --- float (шт) ---
    f_sh = 0.0
    try:
        if "_float_shares" in globals():
            f_sh = float(max(0.0, globals()["_float_shares"](sym) or 0.0))
        else:
            import yfinance as yf

            t = yf.Ticker(sym)
            fi = getattr(t, "fast_info", None)
            if fi:
                try:
                    fs = getattr(fi, "shares_float", None)
                    if fs is None and isinstance(fi, dict):
                        fs = fi.get("shares_float")
                    if fs:
                        f_sh = float(fs)
                except Exception:
                    pass
            if not f_sh:
                try:
                    info = t.get_info(retries=1)
                    fs = info.get("floatShares") or info.get("sharesFloat")
                    if fs:
                        f_sh = float(fs)
                except Exception:
                    pass
    except Exception:
        f_sh = 0.0
    if f_sh > 0:
        out["float_m"] = round(f_sh / 1e6, 2)

    # --- ADV (среднедневной объём, шт) ---
    adv = 0.0
    try:
        if "_adv" in globals():
            adv = float(max(0.0, globals()["_adv"](sym) or 0.0))
        else:
            import pandas as pd
            import yfinance as yf

            df = yf.download(
                sym, period="45d", interval="1d", progress=False, auto_adjust=False
            )
            if df is not None and not df.empty and "Volume" in df:
                vol = pd.Series(df["Volume"]).dropna().astype(float)
                if len(vol) >= 5:
                    adv = float(vol.tail(20).mean())
    except Exception:
        adv = 0.0

    # --- sharesShort / SI% / DTC из Yahoo ---
    shares_short = 0.0
    try:
        import yfinance as yf

        t = yf.Ticker(sym)
        yfi = t.get_info(retries=1) or {}
        cand = [
            yfi.get(k)
            for k in ("sharesShort", "shortSharesOutstanding", "shortInterest")
        ]
        for v in cand:
            try:
                if v is not None and float(v) > 0:
                    shares_short = float(v)
                    break
            except Exception:
                continue
    except Exception:
        shares_short = 0.0

    si_pct = 0.0
    dtc = 0.0
    try:
        if f_sh > 0 and shares_short > 0:
            si_pct = float(min(100.0, max(0.0, 100.0 * shares_short / f_sh)))
        if adv > 0 and shares_short > 0:
            dtc = float(max(0.0, shares_short / adv))
    except Exception:
        pass

    # --- FINVIZ fallback, если включено и нужны подстановки ---
    try:
        if os.getenv("ELIOS_SQ_USE_FINVIZ", "1") == "1" and (
            "_finviz_short" in globals()
        ):
            if (si_pct == 0.0) or (dtc == 0.0):
                fv_si, fv_dtc = globals()["_finviz_short"](sym)
                if (si_pct == 0.0) and isinstance(fv_si, (int, float)) and fv_si > 0:
                    si_pct = float(min(100.0, fv_si))
                if (dtc == 0.0) and isinstance(fv_dtc, (int, float)) and fv_dtc > 0:
                    dtc = float(fv_dtc)
    except Exception:
        pass

    # --- Fee% — iBorrowDesk fallback ---
    fee_pct = 0.0
    try:
        if os.getenv("ELIOS_SQ_USE_IBD", "1") == "1" and (
            "_iborrowdesk_fee" in globals()
        ):
            f = globals()["_iborrowdesk_fee"](sym)
            if isinstance(f, (int, float)) and f > 0:
                fee_pct = float(f)
    except Exception:
        fee_pct = 0.0

    # --- Utilization (оценка дневной ротации флоата) ---
    util_pct = 0.0
    try:
        dv = float(day_volume or 0.0)
        if dv <= 0 and adv > 0:
            dv = adv  # если не дали факта за день — возьмём ADV как прокси
        if f_sh > 0 and dv > 0:
            util_pct = float(min(100.0, 100.0 * dv / f_sh))
    except Exception:
        util_pct = 0.0

    # --- SSR флаг (без онлайна по падению — оставим False) ---
    ssr_flag = False

    # --- Риск-флаги для лонгов + скор ---
    # мягкие пороги, чтобы не "орать" без повода на крупных тикерах
    long_risk = bool(
        (si_pct >= 15.0) or (dtc >= 3.0) or (fee_pct >= 5.0) or (util_pct >= 25.0)
    )
    short_op = bool((si_pct >= 20.0) and (dtc >= 4.0) and (fee_pct >= 7.0))

    # Сводный скор 0..100, с мягкими кепами
    score = 0.0
    try:
        w_si, w_dtc, w_fee, w_util = 0.40, 0.40, 0.10, 0.10
        s_si = min(30.0, si_pct) / 30.0  # 0..1
        s_dtc = min(10.0, dtc) / 10.0
        s_fee = min(50.0, fee_pct) / 50.0
        s_util = min(50.0, util_pct) / 50.0
        score = float(
            100.0 * (w_si * s_si + w_dtc * s_dtc + w_fee * s_fee + w_util * s_util)
        )
    except Exception:
        score = 0.0

    out.update(
        {
            "squeeze_score": round(score, 2),
            "si_pct": float(round(si_pct, 2)),
            "dtc": float(round(dtc, 2)),
            "float_m": float(round(out.get("float_m", 0.0), 2)),
            "fee_pct": float(round(fee_pct, 2)),
            "util_pct": float(round(util_pct, 2)),
            "ssr_flag": bool(ssr_flag),
            "is_squeeze_long_risk": bool(long_risk),
            "is_squeeze_short_opportunity": bool(short_op),
        }
    )

    try:
        _save_cache(sym, out)
    except Exception:
        pass
    return out


def _yahoo_short_interest(symbol: str, timeout=8):
    """
    Используем публичный endpoint:
      https://query1.finance.yahoo.com/v10/finance/quoteSummary/{SYM}?modules=defaultKeyStatistics
    Достаём:
      - si_pct: defaultKeyStatistics.shortPercentOfFloat.raw * 100
      - dtc:    defaultKeyStatistics.shortRatio.raw
    Возвращает dict или None.
    """
    import requests

    sym = str(symbol).upper().strip()
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
    params = {"modules": "defaultKeyStatistics"}
    hdr = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=hdr, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json() or {}
        res = (((j.get("quoteSummary") or {}).get("result") or []) or [None])[0] or {}
        ks = res.get("defaultKeyStatistics") or {}

        def _g(d, *path):
            cur = d
            for k in path:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k)
            return cur

        spof = _g(ks, "shortPercentOfFloat", "raw")
        sratio = _g(ks, "shortRatio", "raw")
        out = {}
        if isinstance(spof, (int, float)):
            out["si_pct"] = float(spof) * 100.0
        if isinstance(sratio, (int, float)):
            out["dtc"] = float(sratio)
        return out if out else None
    except Exception:
        return None


# ---------- yfinance.get_info() short interest (free) ----------
def _yfi_short_interest(symbol: str):
    """
    Пытаемся достать SI%/DTC из yfinance.Ticker.get_info():
      - shortPercentOfFloat -> si_pct
      - shortRatio         -> dtc
      - (fallback) sharesShort/floatShares -> si_pct
      - (fallback) sharesShort/ADV10D      -> dtc
    """
    try:
        import yfinance as yf

        t = yf.Ticker(str(symbol).upper())
        info = t.get_info(retries=1) or {}
        out = {}

        # прямые поля
        spof = info.get("shortPercentOfFloat")
        if isinstance(spof, (int, float)):
            out["si_pct"] = float(spof) * 100.0
        sratio = info.get("shortRatio")
        if isinstance(sratio, (int, float)):
            out["dtc"] = float(sratio)

        # расчётные поля, если прямые пусты
        shares_short = info.get("sharesShort")
        float_shares = info.get("floatShares") or info.get("sharesFloat")
        adv = info.get("averageDailyVolume10Day") or info.get("averageDailyVolume")

        # SI% = sharesShort / floatShares * 100
        if (
            (not isinstance(out.get("si_pct"), (int, float)))
            and isinstance(shares_short, (int, float))
            and isinstance(float_shares, (int, float))
            and float_shares > 0
        ):
            out["si_pct"] = float(shares_short) / float(float_shares) * 100.0

        # DTC = sharesShort / ADV
        if (
            (not isinstance(out.get("dtc"), (int, float)))
            and isinstance(shares_short, (int, float))
            and isinstance(adv, (int, float))
            and adv > 0
        ):
            out["dtc"] = float(shares_short) / float(adv)

        return out if out else None
    except Exception:
        return None


def _marketwatch_short_interest(symbol: str):
    """
    Free fallback: парсим MarketWatch /investing/stock/<symbol> на:
      - Short interest as % of shares outstanding  -> si_pct
      - Short interest ratio (days to cover)       -> dtc
    """
    try:
        import re

        import requests

        url = f"https://www.marketwatch.com/investing/stock/{str(symbol).lower()}"
        hdr = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml",
        }
        r = requests.get(url, headers=hdr, timeout=12)
        if r.status_code != 200 or not r.text:
            return None
        html = r.text

        def _num(x):
            try:
                return float(str(x).replace(",", "").strip())
            except Exception:
                return None

        si_pct = None
        # "Short interest as % of shares outstanding"
        m = re.search(
            r"Short interest as % of shares outstanding.*?<span[^>]*>([\d\.,]+)\s*%",
            html,
            re.I | re.S,
        )
        if m:
            si_pct = _num(m.group(1))
        dtc = None
        # "Short interest ratio (days to cover)"
        m = re.search(
            r"Short interest ratio.*?</small>\s*<span[^>]*>([\d\.,]+)",
            html,
            re.I | re.S,
        )
        if m:
            dtc = _num(m.group(1))
        out = {}
        if si_pct is not None:
            out["si_pct"] = si_pct
        if dtc is not None:
            out["dtc"] = dtc
        return out or None
    except Exception:
        return None


def _yahoo_quote_summary(symbol: str):
    # Yahoo quoteSummary fallback: пытаемся достать sharesShort / shortRatio / shortPercentOfFloat
    try:
        import requests

        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "defaultKeyStatistics,summaryDetail"}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = requests.get(url, params=params, headers=headers, timeout=12)
        if r.status_code != 200:
            return None
        j = r.json() or {}
        res = (j.get("quoteSummary") or {}).get("result") or []
        if not res:
            return None
        ks = res[0].get("defaultKeyStatistics") or {}

        def _raw(d, k):
            v = d.get(k)
            if isinstance(v, dict):
                v = v.get("raw")
            return v

        out = {}
        spf = _raw(ks, "shortPercentOfFloat")  # доля от float (0.xx)
        if isinstance(spf, (int, float)) and spf > 0:
            out["si_pct"] = float(spf) * 100.0

        sr = _raw(ks, "shortRatio")  # дни покрытия (DTC)
        if isinstance(sr, (int, float)) and sr > 0:
            out["dtc"] = float(sr)

        ss = _raw(ks, "sharesShort")  # абсолютное кол-во шортов
        if isinstance(ss, (int, float)) and ss > 0:
            out["shares_short"] = float(ss)

        return out or None
    except Exception:
        return None
