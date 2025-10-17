[Unit]
Description=Elios Trade Executor (prep_signals -> trade_executor)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/root/stockbot
Environment=PYTHONPATH=/root/stockbot
EnvironmentFile=-/root/stockbot/.env
ExecStart=/bin/bash -lc '/root/stockbot/bin/execute_trades.sh'
