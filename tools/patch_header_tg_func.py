# -*- coding: utf-8 -*-
from __future__ import annotations

import py_compile
import re
from pathlib import Path

fp = Path("core/trading/signal_engine.py")
src = fp.read_text(encoding="utf-8", errors="ignore")
lines = src.splitlines(True)

# Найти блок, начинающийся с комментария "Календарь рынка"
start = None
for i, ln in enumerate(lines):
    if re.search(r"Календар[ья] рынка", ln):
        start = i
        break

if start is None:
    print("❌ Не нашёл маркер 'Календарь рынка' — патч не применён.")
    raise SystemExit(1)

# Найти определение _tg_new_signal и конец «битого» блока (последний pass/except сразу после него)
def_idx = None
for j in range(start, min(len(lines), start + 200)):
    if re.match(r"^\s*def\s+_tg_new_signal\s*\(", lines[j]):
        def_idx = j
        break

if def_idx is None:
    print("❌ Не нашёл def _tg_new_signal — патч не применён.")
    raise SystemExit(1)

# Ищем окончание проблемного блока — пройдём после def и соберём до первой пустой строки,
# затем до строки, где начинается следующий блок (например, ещё один 'except' завершает сломанный try).
end = def_idx
# дойдём до строки с 'pass' в except и дальше ещё одну строку
pass_line = None
for k in range(def_idx, min(len(lines), def_idx + 120)):
    if re.match(r"^\s*pass\s*$", lines[k]):
        pass_line = k
# Если нашли 'pass', возьмём следующую строку как конец; иначе аккуратно ограничим 120 строками
end = (pass_line + 1) if pass_line is not None else min(len(lines), def_idx + 120)

# Сконструируем правильный фрагмент
fixed_block = []
fixed_block.append("# Календарь рынка (мягкий импорт — не ломаемся, если файла нет)\n")
fixed_block.append("try:\n")
fixed_block.append("    from core.utils.market_calendar import is_market_open_today\n")
fixed_block.append("except Exception:\n")
fixed_block.append("    def is_market_open_today():\n")
fixed_block.append("        return True\n")
fixed_block.append("\n")
fixed_block.append("# --- Единый эмиттер алёртов в Telegram в требуемом формате\n")
fixed_block.append("def _tg_new_signal(\n")
fixed_block.append("    symbol: str,\n")
fixed_block.append("    price: float,\n")
fixed_block.append("    percent_change: float,\n")
fixed_block.append("    rsi: float,\n")
fixed_block.append("    ema_dev: float,\n")
fixed_block.append("    atr_pct: float,\n")
fixed_block.append("    volatility_pct: float,\n")
fixed_block.append('    gpt_reply: str = ""\n')
fixed_block.append(") -> None:\n")
fixed_block.append('    """\n')
fixed_block.append("    Формат сообщения:\n")
fixed_block.append("      📊 Новый сигнал (BUY)\n")
fixed_block.append("      📌 $TICKER @ 123.45\n")
fixed_block.append("      ∆%=4.49% | RSI=68.44 | EMA dev=2.92%\n")
fixed_block.append("      ATR%=2.23 | Vol=1.52%\n")
fixed_block.append('      🤖 GPT: "…"\n')
fixed_block.append('    """\n')
fixed_block.append("    try:\n")
fixed_block.append("        from core.utils.telegram import send_telegram_message\n")
fixed_block.append("        msg = (\n")
fixed_block.append('            f"📊 Новый сигнал (BUY)\\n"\n')
fixed_block.append('            f"📌 ${symbol} @ {price:.2f}\\n"\n')
fixed_block.append(
    '            f"∆%={percent_change:.2f}% | RSI={rsi:.2f} | EMA dev={ema_dev:.2f}%\\n"\n'
)
fixed_block.append('            f"ATR%={atr_pct:.2f} | Vol={volatility_pct:.2f}%\\n"\n')
fixed_block.append('            f"🤖 GPT: \\"{gpt_reply}\\""\n')
fixed_block.append("        )\n")
fixed_block.append("        send_telegram_message(msg)\n")
fixed_block.append("    except Exception as e:\n")
fixed_block.append('        print(f"[WARN] Telegram unified signal msg: {e}")\n')
fixed_block.append("\n")

# Применим замену и сохраним бэкап
bak = fp.with_suffix(".py.before_header_fix.bak")
bak.write_text(src, encoding="utf-8")

new_src = "".join(lines[:start]) + "".join(fixed_block) + "".join(lines[end:])
fp.write_text(new_src, encoding="utf-8")

# Проверим синтаксис
try:
    py_compile.compile(str(fp), doraise=True)
    print("✅ Патч применён, syntax OK")
except Exception as e:
    print("❗ После патча всё ещё синтаксическая ошибка:", e)
    print("   Покажи контекст: nl -ba core/trading/signal_engine.py | sed -n '1,200p'")
