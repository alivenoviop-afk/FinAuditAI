#!/usr/bin/env python3
"""Клонирует FinRepGYM (проверщик) и запускает его собственную проверку.

Использование:  python scripts/setup_checker.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "finrepgym"
REPO = "https://github.com/alivenoviop-afk/FinRepGYM"


def main() -> int:
    if not (DEST / ".git").exists():
        print(f"Клонирую {REPO} -> {DEST}")
        r = subprocess.run(["git", "clone", "--depth", "1", REPO, str(DEST)])
        if r.returncode != 0:
            print("Не удалось склонировать. Проверь доступ к git/GitHub.")
            return 1
    else:
        print(f"Уже склонирован: {DEST}")

    print("\nЗапуск проверки FinRepGYM (scripts/check.py)...")
    r = subprocess.run([sys.executable, "scripts/check.py"], cwd=DEST)
    print("\nГотово. Теперь в UI: вкладка «Проверка FinRepGYM».")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
