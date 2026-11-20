# FinAuditAI — платформа обучения ИИ под финансовые аудиты

Платформа для **выбора модели, её тренировки и прогона файлов** через неё.
Локальные и API-модели переключаются в одном UI — медицинская/аудит-компания сама выбирает, что подключать.

**Проверщик качества моделей:** [FinRepGYM](https://github.com/alivenoviop-afk/FinRepGYM) — RL-среда для дообучения
моделей финансового аудита. Каждая обученная/подключённая модель прогоняется через её эпизоды и получает
метрики P/R/F1.

## Возможности

| Функция | Описание |
| --- | --- |
| **Каталог моделей** | Локальные (Ollama, HuggingFace) и API (OpenAI, Claude, любой OpenAI-совместимый) — подключение своего эндпоинта/ключа из UI |
| **Тренировка** | Fine-tuning (LoRA/PEFT) на размеченных аудит-кейсах, фоновые задачи, логи в реальном времени |
| **Прогон файлов** | Загрузка PDF/XLSX/CSV/JSONL/DOCX → парсинг → чанкинг → инференс → структурированный аудит-отчёт |
| **Проверка (FinRepGYM)** | Прогон модели через эпизоды среды с подсчётом P/R/F1 и награды |
| **История** | Все прогоны, отчёты и метрики сохраняются в БД |

## Быстрый старт

```bash
docker compose up --build          # backend :8000, frontend :5173
# или без Docker:
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
```

Открыть: http://localhost:8000 (или http://localhost:5173 в Docker-режиме).

## Архитектура

См. [docs/architecture.md](docs/architecture.md).

```
FinAuditAI/
├── backend/                 # Python — ядро платформы
│   ├── app/
│   │   ├── main.py          # FastAPI: API + раздача веб-UI
│   │   ├── api/             # models / training / inference / files / eval
│   │   ├── core/            # реестр моделей, тренер, пайплайн, evaluator (FinRepGYM)
│   │   ├── providers/       # единый интерфейс: local (Ollama/HF) + api (OpenAI-совм.)
│   │   └── db/              # SQLite/PostgreSQL: модели, задачи, прогоны
│   └── workers/             # фоновые задачи (обучение, большие прогоны)
├── frontend/                # веб-интерфейс (Vite + React)
├── data/
│   ├── datasets/            # размеченные аудит-кейсы для тренировки
│   ├── uploads/             # загруженные файлы
│   └── finrepgym/           # копия FinRepGYM (клонируется скриптом)
└── docs/architecture.md
```

## Связь с FinRepGYM

```bash
python scripts/setup_checker.py   # клонирует FinRepGYM в data/finrepgym (нужен git)
```

После этого в UI появляется кнопка **«Проверка FinRepGYM»** — модель получает задачи среды
(`list_records`, `get_record`, `submit`...) и оценивается по `reward = clip(2*F1 - 1 - 0.002*steps, -1, 1)`.

## Стек

Python 3.11+ · FastAPI · SQLAlchemy · Celery/Redis (опц.) · HuggingFace/PEFT · Ollama · React + Vite · Docker Compose
