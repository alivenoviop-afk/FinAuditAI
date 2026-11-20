"""Интеграция с FinRepGYM — проверщик качества обученной модели.

Репозиторий-проверщик: https://github.com/alivenoviop-afk/FinRepGYM
Среда даёт агенту действия list_records / get_record / get_document / calculator / submit,
модель играет роль политики, а награда считается как clip(2*F1 - 1 - 0.002*steps, -1, 1).

Здесь:
  1) setup_checker() — клонирует FinRepGYM в data/finrepgym;
  2) evaluate_model() — прогоняет модель через эпизоды среды и считает P/R/F1.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.core.config import settings
from app.providers.base import ChatMessage, ProviderError
from app.providers.registry import get_provider


def setup_checker() -> dict:
    """Клонирует FinRepGYM (если ещё не склонирован) и запускает его проверку."""
    dest = Path(settings.finrepgym_dir)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", settings.finrepgym_repo, str(dest)],
            check=True, capture_output=True, text=True, timeout=300,
        )
    # Единая проверка среды: python scripts/check.py
    proc = subprocess.run(
        [sys.executable, "scripts/check.py"], cwd=dest,
        capture_output=True, text=True, timeout=600,
    )
    ok = proc.returncode == 0
    return {
        "path": str(dest),
        "ok": ok,
        "stdout": (proc.stdout or "")[-4000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


def _episode_files(episode_dir: Path) -> dict[str, str]:
    """Читает открытые файлы эпизода (видны агенту): manifest, ledger, invoices..."""
    files = {}
    for name in ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                 "fx.csv", "policy.json"):
        p = episode_dir / "open" / name
        if p.exists():
            files[name] = p.read_text(encoding="utf-8", errors="replace")[:6000]
    return files


def _episode_truth(episode_dir: Path) -> list[dict]:
    truth_path = episode_dir / "closed" / "truth.json"
    if not truth_path.exists():
        return []
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    if isinstance(truth, dict):
        return truth.get("errors", truth.get("issues", []))
    return truth


def _ask_policy(prov_key: str, model: str, episode_dir: Path, budget: int = 100) -> list[dict]:
    """Политика модели: даём файлы эпизода и просим вернуть находки (submit)."""
    files = _episode_files(episode_dir)
    if not files:
        return []
    prompt = (
        "Ты — ИИ-аудитор. Переданы файлы эпизода аудита:\n\n"
        + "\n\n".join(f"=== {k} ===\n{v}" for k, v in files.items())
        + "\n\nНайди финансовые ошибки (duplicate_entry, period_cutoff, partial_allocation, fx_rate). "
        + 'Отвечай строго JSON: {"findings": [{"type": ..., "affected_ids": [...], '
        + '"amount": ..., "currency": ...}]}'
    )
    prov = get_provider(prov_key)
    out = prov.run(model, [
        ChatMessage("system", "Ты аудитор. Отвечай только валидным JSON."),
        ChatMessage("user", prompt),
    ], max_tokens=2048)
    try:
        m = json.loads(out[out.find("{"): out.rfind("}") + 1])
        findings = m.get("findings", [])
        return findings if isinstance(findings, list) else []
    except (ValueError, json.JSONDecodeError):
        return []


def _match(truth: list[dict], findings: list[dict]) -> tuple[int, int, int]:
    """Упрощённый matcher: совпадение по типу и множеству affected_ids (как в FinRepGYM)."""
    def norm(e: dict) -> tuple:
        ids = tuple(sorted(str(i) for i in e.get("affected_ids", [])))
        return (str(e.get("type", "")), ids)

    truth_norm = [norm(t) for t in truth]
    used = set()
    tp = 0
    for f in findings:
        nf = norm(f)
        for i, t in enumerate(truth_norm):
            if i not in used and t == nf:
                used.add(i)
                tp += 1
                break
    fp = len(findings) - tp
    fn = len(truth) - tp
    return tp, fp, fn


def evaluate_model(provider_key: str, model: str, split: str = "dev", max_episodes: int = 10) -> dict:
    """Прогоняет модель через эпизоды FinRepGYM и считает P/R/F1/reward."""
    dest = Path(settings.finrepgym_dir)
    if not dest.exists():
        raise ProviderError("FinRepGYM не склонирован. Нажми «Настроить проверщик» или запусти "
                            "scripts/setup_checker.py")

    episodes_dir = dest / "episodes" / split
    if not episodes_dir.exists():
        raise ProviderError(f"Нет папки эпизодов: {episodes_dir}")

    episodes = sorted([p for p in episodes_dir.iterdir() if p.is_dir()])[:max_episodes]
    if not episodes:
        raise ProviderError("Эпизоды не найдены: сначала python scripts/build_episodes.py")

    tp_total = fp_total = fn_total = 0
    rewards: list[float] = []
    per_episode: list[dict] = []

    for ep in episodes:
        truth = _episode_truth(ep)
        findings = _ask_policy(provider_key, model, ep)
        tp, fp, fn = _match(truth, findings)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        reward = max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * len(findings)))
        rewards.append(reward)
        per_episode.append({
            "episode": ep.name, "precision": round(p_, 3), "recall": round(r_, 3),
            "f1": round(f1, 3), "reward": round(reward, 3),
            "truth": len(truth), "findings": len(findings),
        })

    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "model": model,
        "provider": provider_key,
        "split": split,
        "episodes": len(episodes),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        "steps": round(sum(len(e["findings"]) for e in per_episode) / max(1, len(per_episode)), 1),
        "per_episode": per_episode,
    }
