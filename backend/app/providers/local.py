"""Локальные провайдеры: Ollama (инференс) и HuggingFace (LoRA fine-tuning)."""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from app.core.config import settings
from app.providers.base import ChatMessage, ModelInfo, ModelProvider, ProviderError


class OllamaProvider(ModelProvider):
    """Ollama — локальный инференс без сети (http://localhost:11434)."""

    name = "ollama"
    kind = "local"

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def list_models(self) -> list[ModelInfo]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Ollama не запущена (нужен `ollama serve`): {e}") from e
        out = []
        for m in r.json().get("models", []):
            out.append(ModelInfo(
                name=m["name"], provider=self.name, kind=self.kind,
                can_train=False, meta={"size": m.get("size")},
            ))
        return out

    def run(self, model: str, messages: list[ChatMessage], max_tokens: int = 2048) -> str:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
        except Exception as e:
            raise ProviderError(f"Ollama chat error: {e}") from e
        return r.json().get("message", {}).get("content", "")

    def health(self) -> bool:
        try:
            return httpx.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except Exception:
            return False


class HuggingFaceProvider(ModelProvider):
    """HuggingFace — локальный инференс + LoRA fine-tuning (PEFT).

    Инференс идёт через transformers; обучение — классический LoRA-SFT
    по JSONL-датасету {"input": ..., "output": ...}.
    """

    name = "huggingface"
    kind = "local"

    _loaded: dict = {}

    def list_models(self) -> list[ModelInfo]:
        # Модели = базовые из настроек + уже обученные чекпоинты в data/checkpoints
        out = [
            ModelInfo(name="Qwen/Qwen2.5-1.5B-Instruct", provider=self.name, kind=self.kind,
                      can_train=True, base_model="Qwen/Qwen2.5-1.5B-Instruct"),
            ModelInfo(name="mistralai/Mistral-7B-Instruct-v0.3", provider=self.name, kind=self.kind,
                      can_train=True, base_model="mistralai/Mistral-7B-Instruct-v0.3"),
        ]
        ckpt_dir = Path(settings.checkpoints_dir)
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.iterdir()):
                if p.is_dir():
                    out.append(ModelInfo(name=str(p.name), provider=self.name, kind=self.kind,
                                         can_train=False, base_model=str(p),
                                         meta={"checkpoint": True}))
        return out

    def _resolve(self, model: str):
        """Ленивая загрузка модели: чекпоинт или базовая HF-модель."""
        if model in self._loaded:
            return self._loaded[model]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ProviderError("transformers/torch не установлены: pip install -r requirements.txt") from e

        path = str(model)
        ckpt = Path(settings.checkpoints_dir) / model
        if ckpt.exists():
            path = str(ckpt)
        try:
            tok = AutoTokenizer.from_pretrained(path)
            model_obj = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except Exception as e:
            raise ProviderError(f"Не удалось загрузить модель {model}: {e}") from e
        self._loaded[model] = (tok, model_obj)
        return self._loaded[model]

    def run(self, model: str, messages: list[ChatMessage], max_tokens: int = 2048) -> str:
        tok, model_obj = self._resolve(model)
        try:
            inputs = tok.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                add_generation_prompt=True, return_tensors="pt",
            ).to(model_obj.device)
            out = model_obj.generate(
                inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        except Exception as e:
            raise ProviderError(f"Inference error: {e}") from e

    def can_train(self, model: str) -> bool:
        return True

    def train(self, model: str, dataset_path: str, on_log=None, on_progress=None) -> dict:
        """LoRA fine-tuning. Оффлайн-датасет: JSONL с полями input/output."""
        def log(msg: str):
            if on_log:
                on_log(msg)

        log(f"Загрузка датасета {dataset_path}")
        rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ProviderError("Датасет пуст")

        try:
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        except ImportError as e:
            raise ProviderError("peft/transformers/datasets не установлены") from e

        log(f"Примеров: {len(rows)}. Загрузка базовой модели {model}")
        tok = AutoTokenizer.from_pretrained(model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model)

        def fmt(r):
            text = f"### Задача:\n{r['input']}\n### Ответ:\n{r['output']}{tok.eos_token}"
            enc = tok(text, truncation=True, max_length=2048, padding="max_length")
            enc["labels"] = enc["input_ids"].copy()
            return enc

        ds = Dataset.from_list(rows).map(fmt, remove_columns=["input", "output"])
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
        model_lora = get_peft_model(base, lora)

        out_dir = str(Path(settings.checkpoints_dir) / f"{model.split('/')[-1]}-lora-{int(__import__('time').time())}")
        args = TrainingArguments(
            output_dir=out_dir, per_device_train_batch_size=1, num_train_epochs=2,
            learning_rate=2e-4, logging_steps=5, save_strategy="epoch",
            report_to=[], remove_unused_columns=False,
        )

        class CB(Trainer):
            def log(self, logs):
                log(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items()))
                if on_progress and "loss" in logs:
                    on_progress(min(0.99, 0.5))
                super().log(logs)

        trainer = CB(model=model_lora, args=args, train_dataset=ds, processing_class=tok)
        log("Старт обучения (LoRA)")
        result = trainer.train()
        trainer.save_model(out_dir)
        tok.save_pretrained(out_dir)
        log(f"Готово. Чекпоинт: {out_dir}")
        if on_progress:
            on_progress(1.0)
        return {"checkpoint": out_dir, "train_loss": float(result.training_loss), "examples": len(rows)}

    def health(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False
