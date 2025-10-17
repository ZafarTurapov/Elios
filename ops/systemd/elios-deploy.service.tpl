[Unit]
Description=Elios auto-deploy (git pull + deps if needed)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/root/stockbot
ExecStart=/bin/bash -lc '/root/stockbot/bin/deploy.sh'
