# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import requests
import sys

OPENAI_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "12"))
DEBUG = os.getenv("ELIOS_GPT_DEBUG", "0") == "1"


def ascii_safe(x) -> str:
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""


def _headers():
    key = os.getenv("OPENAI_API_KEY", "")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _mask(token: str) -> str:
    if not token:
        return ""
    return token[:6] + "..." + token[-4:] if len(token) > 10 else "***"


def gpt_available() -> bool:
    enabled = os.getenv("ELIOS_GPT_GUARD", "1") != "0"
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return enabled and has_key


def diag_status() -> dict:
    """Возвращает диагностическую сводку по доступности GPT."""
    enabled = os.getenv("ELIOS_GPT_GUARD", "1") != "0"
    key = os.getenv("OPENAI_API_KEY", "")
    base = OPENAI_BASE
    model = OPENAI_MODEL
    return {
        "enabled": enabled,
        "has_key": bool(key),
        "key_mask": _mask(key),
        "base": base,
        "model": model,
    }


def selftest() -> dict:
    """Мини-тест запроса к API с безопасным промптом. Печатает диагностический лог при DEBUG."""
    if not gpt_available():
        info = diag_status()
        msg = "disabled" if not info["enabled"] else "no_key"
        if DEBUG:
            print(f"[GPT] selftest -> {msg} {info}", file=sys.stderr, flush=True)
        return {"ok": False, "reason": msg, "diag": info}

    try:
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": 'Return JSON {"verdict":"ALLOW","confidence":1,"rationale":"ok"}',
                },
                {"role": "user", "content": "ping"},
            ],
            "temperature": 0.0,
            "max_tokens": 20,
            "response_format": {"type": "json_object"},
        }
        r = requests.post(
            f"{OPENAI_BASE}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        js = r.json()
        out = json.loads(js["choices"][0]["message"]["content"])
        ok = isinstance(out, dict) and "verdict" in out
        if DEBUG:
            print(
                f"[GPT] selftest http={r.status_code} ok={ok} resp={out}",
                file=sys.stderr,
                flush=True,
            )
        return {"ok": ok, "resp": out}
    except Exception as e:
        if DEBUG:
            print(f"[GPT] selftest error: {e}", file=sys.stderr, flush=True)
        return {"ok": False, "reason": "error:" + ascii_safe(e)}


def evaluate(symbol: str, feats: dict) -> dict:
    """
    Возвращает: {"verdict":"ALLOW"|"DENY","confidence":0..1,"rationale":"..."}.
    При отключении/ошибке: ALLOW с rationale='gpt_disabled'|'gpt_no_key'|'gpt_error:...'
    """
    if os.getenv("ELIOS_GPT_GUARD", "1") == "0":
        return {"verdict": "ALLOW", "confidence": 0.0, "rationale": "gpt_disabled"}
    if not os.getenv("OPENAI_API_KEY"):
        return {"verdict": "ALLOW", "confidence": 0.0, "rationale": "gpt_no_key"}

    brief = {
        "body_pct": feats.get("body_pct"),
        "gap_pct": feats.get("gap_pct"),
        "vol_ratio": feats.get("vol_ratio"),
        "atr_pct": feats.get("atr_pct"),
        "volatility": feats.get("volatility"),
        "rs": feats.get("rs"),
        "short_float_pct": feats.get("short_float_pct"),
        "days_to_cover": feats.get("days_to_cover"),
        "borrow_fee_pct": feats.get("borrow_fee_pct"),
        "upper_wick_pct": feats.get("upper_wick_pct"),
        "lower_wick_pct": feats.get("lower_wick_pct"),
        "rt_price": feats.get("rt_price") or feats.get("close"),
        "vwap": feats.get("vwap"),
        "orh": feats.get("orh"),
        "orl": feats.get("orl"),
        "why": feats.get("why"),
        "score": feats.get("score"),
    }

    system = (
        "Ты строгий ревьюер внутридневных торговых сигналов (Dux-style: системные фильтры + short squeeze). "
        'Ответь КОМПАКТНЫМ JSON: {"verdict":"ALLOW|DENY","confidence":0..1,"rationale":"<=200 симв"}. '
        "Отклоняй при завышенном риске (ATR/волатильность), слабом объёме, если long ниже VWAP/ORH (или short выше VWAP/ORL), "
        "или если свеча ненадёжная (длинные тени). Предпочитай ALLOW при согласованных метриках. Пиши по-русски."
    )
    user = f"Тикер: {symbol}\nМетрики:\n" + json.dumps(brief, ensure_ascii=False)

    try:
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        r = requests.post(
            f"{OPENAI_BASE}/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        js = r.json()
        msg = js["choices"][0]["message"]["content"]
        out = json.loads(msg)
        verdict = str(out.get("verdict", "ALLOW")).upper()
        if verdict not in ("ALLOW", "DENY"):
            verdict = "ALLOW"
        try:
            conf = float(out.get("confidence", 0))
        except Exception:
            conf = 0.0
        rat = str(out.get("rationale", ""))[:400]
        return {
            "verdict": verdict,
            "confidence": max(0.0, min(1.0, conf)),
            "rationale": rat,
        }
    except Exception as e:
        reason = "gpt_error:" + ascii_safe(e)
        if DEBUG:
            print(f"[GPT] evaluate error {symbol}: {e}", file=sys.stderr, flush=True)
        return {"verdict": "ALLOW", "confidence": 0.0, "rationale": reason}
