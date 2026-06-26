# Telegram Notifications — Setup Guide

The bot can send trade signals, stop-outs, and daily P&L summaries to a private
Telegram chat. This is optional — the bot runs normally without it configured.

---

## Step 1 — Create a bot and get its token

1. Open Telegram and search for **@BotFather**.
2. Send the command `/newbot`.
3. Follow the prompts: choose a name (e.g. `My Trading Bot`) and a username
   ending in `bot` (e.g. `mytrading_bot`).
4. BotFather replies with a token that looks like:
   ```
   5555555555:AAGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   Copy this — it is your `TELEGRAM_BOT_TOKEN`.

---

## Step 2 — Find your chat ID

You need the numeric ID of the chat where the bot will send messages.

### Personal chat (recommended for solo traders)

1. Search for your bot's username in Telegram and tap **Start** (or send `/start`).
2. In a browser, open:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
   Replace `<YOUR_TOKEN>` with the token from Step 1.
3. You will see JSON similar to:
   ```json
   {
     "result": [{
       "message": {
         "chat": { "id": 123456789, "type": "private" }
       }
     }]
   }
   ```
4. The value of `"id"` is your `TELEGRAM_CHAT_ID`.

> If the result array is empty, send another message to the bot and refresh the URL.

### Group chat

1. Add your bot to the group.
2. Send a message in the group (e.g. `/start`).
3. Visit the same `getUpdates` URL — the chat `"id"` for a group is a **negative**
   number, e.g. `-987654321`.

---

## Step 3 — Add to .env

Open `.env` in the project root and fill in all six lines:

```
TELEGRAM_BOT_TOKEN=5555555555:AAGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_CHAT_ID=123456789
TELEGRAM_ENABLED=true
ENVIRONMENT=paper
```

### TELEGRAM_ENABLED

| Value | Effect |
|---|---|
| `true` (default) | Notifications are sent normally |
| `false` | All notifications silenced; credentials stay in .env |

Use `TELEGRAM_ENABLED=false` to pause notifications temporarily without
removing your token (useful when debugging the bot without phone spam).

### ENVIRONMENT

| Value | Telegram notifications |
|---|---|
| `paper` (default) | ✅ Enabled |
| `live` | ✅ Enabled |
| `local` | ✅ Enabled |
| `dry_run` | 🔇 Suppressed |
| `test` | 🔇 Suppressed |

Notifications are also automatically suppressed while running under **pytest**
(the `PYTEST_CURRENT_TEST` environment variable that pytest sets per-test is
detected at send time — you do not need to change `.env` to run tests safely).

---

## Step 4 — Send a test message

Run the connectivity test from the project root (venv active):

```bash
python -m bot.telegram_notifier
```

This command always sends a real message regardless of `ENVIRONMENT` or
`TELEGRAM_ENABLED` — its sole purpose is to verify the Telegram connection.
It is the **only** command that intentionally bypasses suppression.

Expected output when configured correctly:

```
  TELEGRAM_BOT_TOKEN : set
  TELEGRAM_CHAT_ID   : set
  TELEGRAM_ENABLED   : true
  ENVIRONMENT        : paper

[PASS] Test message delivered. Check your Telegram app.
```

---

## Suppression summary

Notifications are suppressed when **any** of the following is true:

| Condition | How to trigger |
|---|---|
| Token or chat ID missing | Remove from `.env` |
| `TELEGRAM_ENABLED=false` | Set in `.env` |
| `ENVIRONMENT=test` or `dry_run` | Set in `.env` |
| Running under pytest | Automatic — `PYTEST_CURRENT_TEST` is detected |

Only `python -m bot.telegram_notifier` bypasses these checks.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `getUpdates` returns an empty array | Send a message to the bot first, then refresh |
| HTTP 401 Unauthorized | Token is wrong or has extra whitespace |
| HTTP 400 Bad Request: chat not found | Chat ID is wrong; check that bot was started |
| Message never arrives in group | Bot lacks permission to send messages |
| `[FAIL]` in test but log shows HTTP 200 | Telegram returned `ok: false` — see log for description |
| Notifications stopped mid-session | Check `TELEGRAM_ENABLED` and `ENVIRONMENT` in `.env` |
