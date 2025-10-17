[Unit]
Description=Elios auto-deploy cadence

[Timer]
OnCalendar=*:0/5
Persistent=true
Unit=elios-deploy.service

[Install]
WantedBy=timers.target
