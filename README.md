# FinAuditAI — платформа обучения ИИ под финансовые аудиты

Платформа для **выбора модели, её тренировки и прогона файлов** через неё.
Локальные и API-модели переключаются в одном UI — компания сама выбирает, что подключать.

## Возможности

| Функция | Описание |
| --- | --- |
| **Каталог моделей** | Локальные (Ollama, HuggingFace) и API (OpenAI, Claude, любой OpenAI-совместимый) — подключение своего эндпоинта/ключа из UI |
| **Тренировка** | Fine-tuning (LoRA/PEFT) на размеченных аудит-кейсах, фоновые задачи, логи в реальном времени |
| **Прогон файлов** | Загрузка PDF/XLSX/CSV/JSONL/DOCX → парсинг → чанкинг → инференс → структурированный аудит-отчёт |
| **История** | Все прогоны и отчёты сохраняются в БД |

## Быстрый старт

```bash
docker compose up --build          # backend :8000 с веб-UI
# или без Docker:
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload
```

Открыть: http://localhost:8000

**Облачная версия:** https://finauditai.onrender.com (автодеплой при каждом push в `master`).

## Архитектура

См. [docs/architecture.md](docs/architecture.md).

```
FinAuditAI/
├── backend/                 # Python — ядро платформы
│   ├── app/
│   │   ├── main.py          # FastAPI: API + раздача веб-UI
│   │   ├── api/             # models / training / inference
│   │   ├── core/            # реестр моделей, тренер, пайплайн прогона
│   │   ├── providers/       # единый интерфейс: local (Ollama/HF) + api (OpenAI-совм.)
│   │   └── db/              # SQLite/PostgreSQL: модели, задачи, прогоны
│   └── requirements-ml.txt   # ML-стек (torch/peft) для обучения на GPU
├── frontend/                # веб-интерфейс (plain HTML/JS, без сборки)
├── data/
│   ├── datasets/            # размеченные аудит-кейсы для тренировки
│   ├── uploads/             # загруженные файлы
│   └── checkpoints/         # обученные чекпоинты (LoRA)
└── docs/architecture.md
```

## Стек

Python 3.11+ · FastAPI · SQLAlchemy · HuggingFace/PEFT · Ollama · Docker Compose · GitHub Actions (автодеплой)
