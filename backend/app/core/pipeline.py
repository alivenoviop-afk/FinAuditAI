"""Пайплайн прогона файла через модель: парсинг -> чанки -> инференс -> отчёт."""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.providers.base import ChatMessage, ProviderError
from app.providers.registry import get_provider

SYSTEM_PROMPT = (
    "Ты — аудитор финансовой отчётности. Анализируй переданные данные "
    "(проводки, счета, платежи, курсы) и находь финансовые ошибки.\n"
    "Отвечай СТРОГО валидным JSON без пояснений:\n"
    '{"findings": [{"type": "duplicate_entry|period_cutoff|partial_allocation|fx_rate",'
    ' "affected_ids": [...], "amount": <число>, "currency": "USD|EUR|KZT",'
    ' "evidence": "краткое объяснение"}]}\n'
    "Если ошибок нет — {\"findings\": []}."
)

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 400


# ---------- Парсинг файлов ----------

def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".jsonl", ".json"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        lines = []
        for ws in wb.worksheets:
            lines.append(f"# Лист: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                lines.append(",".join("" if c is None else str(c) for c in row))
        return "\n".join(lines)
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    if suffix == ".docx":
        import docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    raise ProviderError(f"Неподдерживаемый формат файла: {suffix}")


def chunk_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


# ---------- Инференс ----------

def _parse_json(text: str) -> dict:
    # модель может обернуть JSON в ```json ... ```
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"findings": [], "raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"findings": [], "raw": text}


def run_file(provider_key: str, model_name: str, path: Path, max_tokens: int = 2048) -> dict:
    """Полный прогон: читает файл, чанкует, опрашивает модель, собирает отчёт."""
    prov = get_provider(provider_key)
    text = extract_text(path)
    if not text.strip():
        raise ProviderError("Файл пустой или не удалось извлечь текст")
    chunks = chunk_text(text)

    all_findings: list[dict] = []
    raw_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        messages = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", f"Данные для аудита (фрагмент {i + 1}/{len(chunks)}):\n\n{chunk}"),
        ]
        out = prov.run(model_name, messages, max_tokens=max_tokens)
        raw_parts.append(out)
        parsed = _parse_json(out)
        if isinstance(parsed.get("findings"), list):
            all_findings.extend(parsed["findings"])

    return {
        "file": path.name,
        "provider": provider_key,
        "model": model_name,
        "chunks": len(chunks),
        "chars": len(text),
        "findings": all_findings,
        "raw": "\n---\n".join(raw_parts)[:50000],
    }
