# core/trading/test_sell_logic.py
from datetime import datetime

DEFAULT_TAKE_PROFIT = 0.05
DEFAULT_STOP_LOSS = -0.03
EARLY_STOP_LOSS = -0.03  # за 1 день
PARTIAL_TAKE_PROFIT = 0.03  # частичная фиксация

# Симуляция позиций
# added_days — сколько дней держим позицию
test_positions = {
    "AAPL": {"qty": 10, "entry_price": 100, "current_price": 107, "days_held": 5},
    "TSLA": {"qty": 20, "entry_price": 200, "current_price": 190, "days_held": 1},  # early stop
    "MSFT": {"qty": 15, "entry_price": 150, "current_price": 154.5, "days_held": 4},
    "NVDA": {"qty": 5, "entry_price": 400, "current_price": 420, "days_held": 3},
    "AMZN": {"qty": 30, "entry_price": 150, "current_price": 145.5, "days_held": 2}
}

print(f"=== Тест логики SELL ENGINE — {datetime.now().isoformat()} ===")

for symbol, pos in test_positions.items():
    qty = pos["qty"]
    entry = pos["entry_price"]
    current = pos["current_price"]
    days_held = pos["days_held"]

    change_pct = (current - entry) / entry
    pnl = round((current - entry) * qty, 2)
    action = "HOLD"
    reason = None
    qty_to_sell = qty

    # === Логика ===
    if days_held == 1 and change_pct <= EARLY_STOP_LOSS:
        action = "SELL"
        reason = "EARLY_STOP"
    elif change_pct >= DEFAULT_TAKE_PROFIT:
        if change_pct >= PARTIAL_TAKE_PROFIT and change_pct < DEFAULT_TAKE_PROFIT:
            action = "PARTIAL_SELL"
            qty_to_sell = qty // 2
            reason = "PARTIAL_TP_3%"
        else:
            action = "SELL"
            reason = "TAKE_PROFIT_5%"
    elif change_pct <= DEFAULT_STOP_LOSS:
        action = "SELL"
        reason = "STOP_LOSS_-3%"

    print(f"{symbol}: {reason or 'HOLD'} | Δ%={round(change_pct*100,2)} | "
          f"PnL={pnl}$ | SELL={action} | Qty to sell={qty_to_sell}")
