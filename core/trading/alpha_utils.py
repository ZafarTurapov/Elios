# core/trading/alpha_utils.py


def normalize(value, min_val, max_val):
    """Нормализация значения в диапазон 0.0–1.0"""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def calculate_alpha_score(percent_change, volume_ratio, rsi, ema_deviation):
    """
    Расчёт итогового alpha-скоринга.
    Весовые коэффициенты подобраны эмпирически.
    """
    score = (
        0.4 * normalize(percent_change, 0, 10)
        + 0.3 * normalize(volume_ratio, 0, 5)
        + 0.2 * (1.0 if rsi > 70 else 0.0)
        + 0.1 * (1.0 if ema_deviation > 3 else 0.0)
    )
    return round(score, 4)
