"""Daily backup — compress bot data files into a timestamped ZIP.

Usage:
    python scripts/backup.py
    python scripts/backup.py --bot-dir /opt/trading-bot
    python scripts/backup.py --backup-dir /mnt/external/backups --keep 60

Files included (silently skipped if absent):
    trades.csv
    daily_pnl.csv
    trading_bot.log
    position_state.json

Exit codes:
    0  — at least one file backed up successfully
    1  — all source files were missing (nothing backed up)
"""
import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_TARGETS = [
    "trades.csv",
    "daily_pnl.csv",
    "trading_bot.log",
    "position_state.json",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--bot-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Bot project root (default: parent of scripts/)",
    )
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Directory to write ZIP files into (default: <bot-dir>/backups)",
    )
    p.add_argument(
        "--keep",
        type=int,
        default=30,
        help="Number of most-recent backups to retain (default: 30)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bot_dir: Path = args.bot_dir.resolve()
    backup_dir: Path = (args.backup_dir or bot_dir / "backups").resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")
    zip_path = backup_dir / f"backup_{ts}.zip"

    included: list[str] = []
    missing: list[str] = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in _TARGETS:
            src = bot_dir / name
            if src.exists():
                zf.write(src, arcname=name)
                included.append(name)
            else:
                missing.append(name)

    if not included:
        zip_path.unlink(missing_ok=True)
        print(
            "ERROR: no files were backed up — all source files are missing.",
            file=sys.stderr,
        )
        sys.exit(1)

    size_kb = zip_path.stat().st_size / 1024
    print(f"Created: {zip_path}  ({size_kb:.1f} KB)")
    print(f"  Included : {', '.join(included)}")
    if missing:
        print(f"  Skipped  : {', '.join(missing)} (not found)")

    # Prune oldest backups beyond --keep
    all_backups = sorted(backup_dir.glob("backup_*.zip"))
    to_delete = all_backups[: max(0, len(all_backups) - args.keep)]
    for old in to_delete:
        old.unlink()
        print(f"  Pruned   : {old.name}")


if __name__ == "__main__":
    main()
