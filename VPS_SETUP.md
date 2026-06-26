# VPS Setup — Trading Bot on DigitalOcean Ubuntu 24.04

## 1. Create a DigitalOcean Droplet

1. Log in at [digitalocean.com](https://www.digitalocean.com) and click **Create → Droplets**.
2. Choose **Ubuntu 24.04 LTS** as the image.
3. Select a plan — the **Basic $6/mo** (1 vCPU, 1 GB RAM) is enough for this bot.
4. Choose a datacenter region close to US East (New York or Toronto minimize latency to Alpaca).
5. Under **Authentication**, choose **SSH Key**:
   - On your Windows machine, open PowerShell and run:
     ```powershell
     ssh-keygen -t ed25519 -C "trading-bot" -f "$env:USERPROFILE\.ssh\trading_bot"
     ```
   - Copy the public key contents:
     ```powershell
     Get-Content "$env:USERPROFILE\.ssh\trading_bot.pub"
     ```
   - Paste that into the DigitalOcean SSH key field.
6. Give the droplet a hostname like `trading-bot-01` and click **Create Droplet**.
7. Note the droplet's **IPv4 address** once it appears (e.g. `64.225.12.34`).

---

## 2. SSH Login from Windows PowerShell

```powershell
ssh -i "$env:USERPROFILE\.ssh\trading_bot" root@<DROPLET_IP>
```

On first connection, accept the host fingerprint prompt by typing `yes`.

To avoid typing the key path every time, add this block to `~\.ssh\config`:

```
Host trading-bot
    HostName <DROPLET_IP>
    User root
    IdentityFile ~/.ssh/trading_bot
```

Then connect with just:

```powershell
ssh trading-bot
```

---

## 3. Install Python, Git, and venv (on the VPS)

```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git
python3 --version   # should be 3.12.x on Ubuntu 24.04
```

---

## 4. Copy Project Files from Windows to VPS

From your local Windows PowerShell (not the SSH session), copy the project directory.
`rsync` is the cleanest option if you have it via Git Bash or WSL; otherwise use `scp`.

**Option A — rsync (Git Bash / WSL terminal):**
```bash
rsync -av --exclude venv --exclude __pycache__ --exclude "*.pyc" --exclude backups \
  /c/Users/jwieg/trading-bot/ root@<DROPLET_IP>:/opt/trading-bot/
```

**Option B — scp (PowerShell):**
```powershell
scp -i "$env:USERPROFILE\.ssh\trading_bot" -r `
  C:\Users\jwieg\trading-bot `
  root@<DROPLET_IP>:/opt/trading-bot
```

Then SSH in and confirm the files are there:
```bash
ls /opt/trading-bot
# bot/  config.py  requirements.txt  tests/  test_connection.py  ...
```

---

## 5. Create the Virtual Environment and .env

```bash
cd /opt/trading-bot
python3 -m venv venv
source venv/bin/activate
```

Create the `.env` file with your real credentials:

```bash
cat > /opt/trading-bot/.env << 'EOF'
ALPACA_API_KEY=<your_alpaca_api_key>
ALPACA_SECRET_KEY=<your_alpaca_secret_key>
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
TELEGRAM_BOT_TOKEN=<your_telegram_bot_token>
TELEGRAM_CHAT_ID=<your_telegram_chat_id>
TELEGRAM_ENABLED=true
ENVIRONMENT=paper
EOF
```

Lock down the file so only root can read it:

```bash
chmod 600 /opt/trading-bot/.env
```

---

## 6. Install Requirements

```bash
cd /opt/trading-bot
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Run pytest

```bash
cd /opt/trading-bot
source venv/bin/activate
python -m pytest tests/ -v
```

All tests should pass. The test suite mocks Telegram automatically so no real messages are sent.

---

## 8. Run test_connection.py

Verifies Alpaca credentials and live data access:

```bash
cd /opt/trading-bot
source venv/bin/activate
python test_connection.py
```

Expected output ends with:
```
[PASS] Fetch BTC/USD hourly bars (last 3 days)
       rows returned: 72
       latest close : $...

Connection test complete.
```

If any step shows `[FAIL]`, check your `.env` values before proceeding.

---

## 9. Run Telegram Connectivity Test

Sends a real test message to your Telegram chat:

```bash
cd /opt/trading-bot
source venv/bin/activate
python -m bot.telegram_notifier
```

Check your Telegram app — you should receive:
> Trading bot — connectivity test  
> If you see this, Telegram notifications are working correctly.

---

## 10. Create the systemd Service

Create the service unit file:

```bash
cat > /etc/systemd/system/trading-bot.service << 'EOF'
[Unit]
Description=Algorithmic Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trading-bot
EnvironmentFile=/opt/trading-bot/.env
ExecStart=/opt/trading-bot/venv/bin/python -m bot.main
Restart=on-failure
RestartSec=30
StandardOutput=append:/opt/trading-bot/trading_bot.log
StandardError=append:/opt/trading-bot/trading_bot.log

[Install]
WantedBy=multi-user.target
EOF
```

Reload systemd and enable the service to start on boot:

```bash
systemctl daemon-reload
systemctl enable trading-bot
```

---

## 11. Start / Stop / Restart / Status

```bash
# Start the bot
systemctl start trading-bot

# Stop the bot (sends SIGTERM — bot writes final daily P&L on exit)
systemctl stop trading-bot

# Restart (e.g. after updating code)
systemctl restart trading-bot

# Check running status and last few log lines
systemctl status trading-bot
```

---

## 12. Checking Logs

**Live tail** (most useful while testing):
```bash
tail -f /opt/trading-bot/trading_bot.log
```

**Last 100 lines:**
```bash
tail -100 /opt/trading-bot/trading_bot.log
```

**systemd journal** (service start/stop events, crashes):
```bash
journalctl -u trading-bot -n 100 --no-pager
journalctl -u trading-bot -f   # follow
```

**Trade and P&L records:**
```bash
cat /opt/trading-bot/trades.csv
cat /opt/trading-bot/daily_pnl.csv
```

---

## 13. Safe Shutdown

`systemctl stop trading-bot` sends `SIGTERM` to the bot process. The main loop catches this signal, writes a final daily P&L row to `daily_pnl.csv`, cancels any pending stop orders, and exits cleanly.

Wait a few seconds after stopping before inspecting output files to ensure the shutdown handler has completed:

```bash
systemctl stop trading-bot
sleep 5
tail -5 /opt/trading-bot/trading_bot.log
tail -2 /opt/trading-bot/daily_pnl.csv
```

To permanently disable auto-restart on boot:

```bash
systemctl disable trading-bot
systemctl stop trading-bot
```

---

## 14. Automatic Daily Backups

The script `scripts/backup.py` zips the four data files into a timestamped archive and prunes the oldest entries so only the last 30 backups are kept.

**Files backed up:**
- `trades.csv`
- `daily_pnl.csv`
- `trading_bot.log`
- `position_state.json`

Archives are written to `/opt/trading-bot/backups/` with names like `backup_2026-06-26_000500Z.zip`.

### Test the script manually first

```bash
cd /opt/trading-bot
source venv/bin/activate
python scripts/backup.py
```

Expected output:
```
Created: /opt/trading-bot/backups/backup_2026-06-26_000500Z.zip  (12.3 KB)
  Included : trades.csv, daily_pnl.csv, trading_bot.log, position_state.json
```

Inspect the archive:
```bash
ls -lh /opt/trading-bot/backups/
unzip -l /opt/trading-bot/backups/backup_*.zip | tail -1
```

### Create the systemd service unit

```bash
cat > /etc/systemd/system/trading-bot-backup.service << 'EOF'
[Unit]
Description=Trading Bot Daily Backup

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/trading-bot
ExecStart=/opt/trading-bot/venv/bin/python scripts/backup.py
StandardOutput=journal
StandardError=journal
EOF
```

### Create the systemd timer (runs daily at 00:05 UTC)

The 5-minute offset ensures the bot's midnight daily P&L flush has completed before the backup runs.
`Persistent=true` means a missed backup (e.g. server was rebooting at midnight) is run immediately on next boot.

```bash
cat > /etc/systemd/system/trading-bot-backup.timer << 'EOF'
[Unit]
Description=Trading Bot Daily Backup Timer
Requires=trading-bot-backup.service

[Timer]
OnCalendar=*-*-* 00:05:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF
```

### Enable and start the timer

```bash
systemctl daemon-reload
systemctl enable --now trading-bot-backup.timer
```

### Verify the timer is scheduled

```bash
systemctl list-timers trading-bot-backup.timer
```

Output shows the next trigger time and when the last run occurred:
```
NEXT                          LEFT      LAST                          PASSED   UNIT
Thu 2026-06-27 00:05:00 UTC   23h left  Thu 2026-06-26 00:05:01 UTC   42s ago  trading-bot-backup.timer
```

### Check backup logs

```bash
# Output from the last backup run
journalctl -u trading-bot-backup.service -n 20 --no-pager

# Follow live (useful when triggering a manual test run)
journalctl -u trading-bot-backup.service -f
```

### Trigger a backup immediately (without waiting for midnight)

```bash
systemctl start trading-bot-backup.service
journalctl -u trading-bot-backup.service -n 10 --no-pager
```

### List stored backups

```bash
ls -lh /opt/trading-bot/backups/
```

The oldest file is automatically deleted once there are more than 30. At ~10–50 KB per archive, 30 backups use under 2 MB.

### Adjust retention or backup directory

Pass flags directly to the script via the service unit, or run ad hoc:

```bash
# Keep 60 days instead of 30
python scripts/backup.py --keep 60

# Write backups to a different location (e.g. a mounted volume)
python scripts/backup.py --backup-dir /mnt/volume/bot-backups --keep 90
```

To make a flag permanent, edit the `ExecStart` line in `/etc/systemd/system/trading-bot-backup.service` and run `systemctl daemon-reload`.

---

## Updating the Bot After Code Changes

From your Windows PowerShell (rsync approach):
```bash
rsync -av --exclude venv --exclude __pycache__ --exclude "*.pyc" --exclude backups \
  /c/Users/jwieg/trading-bot/ root@<DROPLET_IP>:/opt/trading-bot/
```

Then on the VPS:
```bash
cd /opt/trading-bot
source venv/bin/activate
pip install -r requirements.txt          # only needed if requirements changed
python -m pytest tests/ -v               # confirm nothing broke
systemctl restart trading-bot
systemctl status trading-bot
```
