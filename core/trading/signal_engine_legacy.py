# -*- coding: utf-8 -*-
"""
Лёгкая обёртка: всё выполнение передаётся в актуальный движок core.trading.signal_engine.
Это устраняет старые баги (в т.ч. 'Close' из-за MultiIndex/CSV) и держит единый вход.
"""
from __future__ import annotations


def main():
    # Импортируем «на лету», чтобы избежать циклических импортов при линтерах
    from core.trading.signal_engine import main as new_main

    return new_main()


if __name__ == "__main__":
    main()
