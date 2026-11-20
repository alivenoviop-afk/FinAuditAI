"""Функционал FinRepGYM внутри обучения (пакет audit_env встроен в проект).

Две возможности:
  * ``generate_sft_dataset`` — генерирует эпизоды аудита, прогоняет экспертную
    политику и сохраняет пары «наблюдение → действие» в JSONL
    (``{"input": ..., "output": ...}``) — готовый датасет для LoRA-обучения;
  * ``start_rollout`` — модель сама играет роль аудитора в среде
    (действия list_records / get_record / get_document / calculator / submit),
    считаются метрики P/R/F1 и награда, траектории опционально сохраняются
    датасетом для дообучения.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.audit_env import AuditEnv, ISSUE_TYPES
from app.core.audit_env.env import ACTION_NAMES, POLICY_BRIEF, TASK_TEXT
from app.core.audit_env.policies import OraclePolicy, ReconciliationPolicy
from app.core.config import settings
from app.providers.base import ChatMessage, ProviderError
from app.providers.registry import get_provider

# Пресеты конфигурации эпизодов
PRESETS: dict[str, dict] = {
    "all4": {"issue_types": list(ISSUE_TYPES), "n_issues": 4},
    "default": {"issue_types": ["duplicate_entry", "period_cutoff"], "n_issues": 2},
    "scenarios": {"issue_types": [], "n_issues": 0, "scenarios": ["S1", "S2"]},
    "clean": {"clean": True},
}

MAX_ROLLOUT_EPISODES = 10
MAX_ROLLOUT_STEPS = 30
_HISTORY_LIMIT = 20

_SYSTEM = ("Ты — ИИ-аудитор. Отвечай ТОЛЬКО одним JSON-действием "
           "без пояснений и без markdown-обёртки.")


# --------------------------------------------------------------------------- #
# формат промпта (одинаковый для эксперта и модели — модель обучается на нём)
# --------------------------------------------------------------------------- #
def _format_input(obs: dict, history: list[str], episode_id: str) -> str:
    parts = [
        f"Задача: {TASK_TEXT}",
        "Правила:\n" + "\n".join(f"- {b}" for b in POLICY_BRIEF),
        f"episode_id: {episode_id}",
    ]
    if isinstance(obs.get("manifest"), dict):
        parts.append("Манифест: " + json.dumps(obs["manifest"], ensure_ascii=False))
    if history:
        parts.append("Твои предыдущие действия:\n" + "\n".join(history[-_HISTORY_LIMIT:]))
    parts.append("Наблюдение:\n" + json.dumps(obs, ensure_ascii=False))
    parts.append('Ответь следующим действием в JSON: {"action": "...", ...}')
    return "\n\n".join(parts)


def _parse_action(raw: str, episode_id: str) -> dict | None:
    """Достаёт JSON-действие из ответа модели. None — если разобрать не удалось."""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("action") not in ACTION_NAMES:
        return None
    if obj["action"] == "submit" and obj.get("episode_id") != episode_id:
        # без совпадения episode_id среда отклонит submit — в датасет не идём
        return None
    return obj


def _report_metrics(report: dict) -> tuple[float, float, float, float, int]:
    m = report.get("metrics") or {}
    return (float(m.get("precision") or 0.0), float(m.get("recall") or 0.0),
            float(m.get("f1") or 0.0), float(report.get("true_reward") or 0.0),
            int(report.get("steps_used") or 0))


# --------------------------------------------------------------------------- #
# 1. датасет из экспертных траекторий (SFT)
# --------------------------------------------------------------------------- #
def generate_sft_dataset(episodes: int, seed: int, preset: str,
                         policy: str = "reconciliation") -> dict:
    """Эпизоды + экспертная политика → JSONL {"input", "output"} в data/datasets."""
    if preset not in PRESETS:
        raise ProviderError(f"Неизвестный пресет: {preset}")
    config = dict(PRESETS[preset])
    env = AuditEnv(mode="train")

    samples: list[dict] = []
    f1s: list[float] = []
    rewards: list[float] = []
    steps_list: list[int] = []
    t0 = time.monotonic()

    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i, config=config)
        episode_id = obs["manifest"]["episode_id"]
        expert = OraclePolicy(env.evaluator_truth()) if policy == "oracle" \
            else ReconciliationPolicy()

        history: list[str] = []
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated) and steps < env.max_steps:
            action = expert(obs)
            action_json = json.dumps(action, ensure_ascii=False)
            samples.append({
                "input": _format_input(obs, history, episode_id),
                "output": action_json,
            })
            history.append(action_json)
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1

        p, r, f1, reward, _s = _report_metrics(env.evaluator_report())
        f1s.append(f1)
        rewards.append(reward)
        steps_list.append(_s)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"finrepgym_{preset}_{policy}_{ts}.jsonl"
    path = settings.datasets_dir / name
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    n = max(1, episodes)
    return {
        "ok": True,
        "name": name,
        "episodes": episodes,
        "samples": len(samples),
        "avg_f1": round(sum(f1s) / n, 4),
        "avg_reward": round(sum(rewards) / n, 4),
        "avg_steps": round(sum(steps_list) / n, 1),
        "seconds": round(time.monotonic() - t0, 2),
        "policy": policy,
        "preset": preset,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# 2. прогон модели в среде (rollout) — метрики + опционально датасет
# --------------------------------------------------------------------------- #
def _rollout_one(provider_key: str, model: str, seed: int, config: dict,
                 save_pairs: bool) -> tuple[dict, list[dict]]:
    prov = get_provider(provider_key)
    env = AuditEnv(mode="train", max_steps=MAX_ROLLOUT_STEPS, time_limit_s=600.0)
    obs, _ = env.reset(seed=seed, config=config)
    episode_id = obs["manifest"]["episode_id"]

    history: list[str] = []
    pairs: list[dict] = []
    invalid = 0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < MAX_ROLLOUT_STEPS:
        prompt = _format_input(obs, history, episode_id)
        raw = prov.run(model, [ChatMessage("system", _SYSTEM),
                               ChatMessage("user", prompt)], max_tokens=1024)
        action = _parse_action(raw, episode_id)
        if action is None:
            invalid += 1
            history.append("[модель ответила не тем — среда отклонила действие]")
            # среда даст ограниченную ошибку и потратит шаг — честный отклик
            obs, _reward, terminated, truncated, _info = env.step({"action": raw[:80] or "?"})
            steps += 1
            continue

        action_json = json.dumps(action, ensure_ascii=False)
        if save_pairs:
            pairs.append({"input": prompt, "output": action_json})
        history.append(action_json)
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1

    p, r, f1, reward, steps_used = _report_metrics(env.evaluator_report())
    stats = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
             "reward": round(reward, 4), "steps": steps_used, "invalid": invalid}
    return stats, pairs


def rollout_dataset(provider_key: str, model: str, episodes: int, seed: int,
                    preset: str, save: bool, on_log=None, on_progress=None) -> dict:
    """Прогон модели через эпизоды среды: метрики + (опц.) датасет её траекторий."""
    if preset not in PRESETS:
        raise ProviderError(f"Неизвестный пресет: {preset}")
    config = dict(PRESETS[preset])
    eps = max(1, min(episodes, MAX_ROLLOUT_EPISODES))
    t0 = time.monotonic()

    all_pairs: list[dict] = []
    ps: list[float] = []
    rs: list[float] = []
    f1s: list[float] = []
    rewards: list[float] = []
    steps_list: list[int] = []
    invalid_total = 0

    for i in range(eps):
        stats, pairs = _rollout_one(provider_key, model, seed + i, config, save)
        all_pairs.extend(pairs)
        ps.append(stats["precision"])
        rs.append(stats["recall"])
        f1s.append(stats["f1"])
        rewards.append(stats["reward"])
        steps_list.append(stats["steps"])
        invalid_total += stats["invalid"]
        if on_log:
            on_log(f"Эпизод {i + 1}/{eps}: P={stats['precision']} R={stats['recall']} "
                   f"F1={stats['f1']} reward={stats['reward']} шагов={stats['steps']} "
                   f"отклонено ответов={stats['invalid']}")
        if on_progress:
            on_progress((i + 1) / eps)

    result = {
        "episodes": eps,
        "precision": round(sum(ps) / eps, 4),
        "recall": round(sum(rs) / eps, 4),
        "f1": round(sum(f1s) / eps, 4),
        "reward": round(sum(rewards) / eps, 4),
        "avg_steps": round(sum(steps_list) / eps, 1),
        "invalid": invalid_total,
        "seconds": round(time.monotonic() - t0, 1),
        "dataset": None,
    }

    if save and all_pairs:
        safe_model = re.sub(r"[^\w.-]+", "_", model)[:40]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = f"finrepgym_rollout_{safe_model}_{ts}.jsonl"
        with (settings.datasets_dir / name).open("w", encoding="utf-8") as fh:
            for s in all_pairs:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        result["dataset"] = name
        result["samples"] = len(all_pairs)

    return result


# --------------------------------------------------------------------------- #
# фоновый прогон (задача в БД — как у обучения)
# --------------------------------------------------------------------------- #
def start_rollout(job_id: int) -> None:
    threading.Thread(target=_rollout_job, args=(job_id,), daemon=True).start()


def _rollout_job(job_id: int) -> None:
    from app.db.models import SessionLocal, TrainingJob

    db = SessionLocal()
    job = db.get(TrainingJob, job_id)
    if job is None:
        db.close()
        return
    job.status = "running"
    db.commit()

    def on_log(msg: str) -> None:
        job.logs = (job.logs + msg + "\n")[-20000:]
        db.commit()

    def on_progress(p: float) -> None:
        job.progress = min(0.99, p)
        db.commit()

    try:
        cfg = job.result or {}
        stats = rollout_dataset(
            provider_key=cfg.get("provider", ""),
            model=cfg.get("model", ""),
            episodes=int(cfg.get("episodes", 3)),
            seed=int(cfg.get("seed", 42)),
            preset=cfg.get("preset", "all4"),
            save=bool(cfg.get("save", True)),
            on_log=on_log,
            on_progress=on_progress,
        )
        job.result = {**cfg, **stats}
        job.status = "done"
        job.progress = 1.0
        on_log(f"ИТОГО: F1={stats['f1']} reward={stats['reward']} "
               f"P={stats['precision']} R={stats['recall']} шагов={stats['avg_steps']}"
               + (f" · датасет {stats['dataset']}" if stats.get("dataset") else ""))
    except Exception as e:  # noqa: BLE001 — ошибки уходят в UI
        job.status = "failed"
        on_log(f"ОШИБКА: {e}")
    finally:
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        db.close()


def save_rollout_job(provider: str, model: str, episodes: int, seed: int,
                     preset: str, save: bool) -> int:
    """Создаёт запись задачи прогона и запускает её в фоне. Возвращает id."""
    from app.db.models import SessionLocal, TrainingJob

    db = SessionLocal()
    try:
        job = TrainingJob(
            dataset_path="прогон в среде",
            result={"kind": "gym_rollout", "provider": provider, "model": model,
                    "provider_model": f"{provider}:{model}", "episodes": episodes,
                    "seed": seed, "preset": preset, "save": save},
        )
        db.add(job)
        db.commit()
        job_id = job.id
    finally:
        db.close()
    start_rollout(job_id)
    return job_id
