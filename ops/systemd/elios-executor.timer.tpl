[Unit]
Description=Elios Executor cadence (every 15min)

[Timer]
OnCalendar=*:0/15
Persistent=true
Unit=elios-executor.service

[Install]
WantedBy=timers.target
