#!/usr/bin/env bash
# Return 0 если сейчас в торговом окне, иначе 1.
# Окно: ПН–ПТ 18:30–23:59 и ВТ–СБ 00:00–01:00 (Asia/Tashkent)

set -euo pipefail

# Если TZ уже задаётся в unit — ок. Иначе можно раскомментировать:
# export TZ=Asia/Tashkent

# ТЕСТОВЫЕ оверрайды:
#   FAKE_DOW=1..7 (1=Mon)
#   FAKE_TIME=HH:MM (локальное)
dow="${FAKE_DOW:-$(date +%u)}"
if [[ -n "${FAKE_TIME:-}" ]]; then
  IFS=: read -r hh mm <<<"$FAKE_TIME"
else
  hh="$(date +%H)"
  mm="$(date +%M)"
fi

# минуты с начала суток
time=$((10#$hh*60 + 10#$mm))
in_window=1

# Вечернее окно: ПН–ПТ 18:30..23:59
if [[ $dow -ge 1 && $dow -le 5 ]]; then
  if [[ $time -ge 1110 ]]; then  # 18*60+30
    in_window=0
  fi
fi

# Ночной хвост: ВТ–СБ 00:00..01:00
if [[ $dow -ge 2 && $dow -le 6 ]]; then
  if [[ $time -lt 60 ]]; then
    in_window=0
  fi
fi

if [[ $in_window -eq 0 ]]; then
  exit 0
else
  printf '[within_sell_window] вне окна: dow=%s time=%02d:%02d\n' "$dow" "$hh" "$mm" >&2
  exit 1
fi
