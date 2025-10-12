from datetime import datetime

def is_market_open_today():
    weekday = datetime.now().weekday()  # Пн = 0, ..., Вс = 6
    return weekday < 5  # Пн–Пт → True, Сб–Вс → False
