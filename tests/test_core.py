"""Быстрые тесты ядра: парсинг/чанкинг, matcher, конфиг, API-роуты."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.pipeline import chunk_text, _parse_json  # noqa: E402


class TestPipeline(unittest.TestCase):
    def test_chunk_small(self):
        self.assertEqual(chunk_text("abc"), ["abc"])

    def test_chunk_overlap(self):
        text = "x" * 10000
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 1)
        # перекрытие: размер чанка 4000, overlap 400
        self.assertEqual(len(chunks[0]), 4000)
        self.assertEqual(chunks[1][:400], "x" * 400)

    def test_parse_json_plain(self):
        self.assertEqual(_parse_json('{"findings": []}'), {"findings": []})

    def test_parse_json_fenced(self):
        out = _parse_json('```json\n{"findings": [{"type": "fx_rate"}]}\n```')
        self.assertEqual(len(out["findings"]), 1)

    def test_parse_json_garbage(self):
        out = _parse_json("не json")
        self.assertEqual(out["findings"], [])


class TestAPI(unittest.TestCase):
    def test_health(self):
        from fastapi.testclient import TestClient
        from app.main import app
        c = TestClient(app)
        r = c.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn("providers", r.json())

    def test_models_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        c = TestClient(app)
        r = c.get("/api/models")
        self.assertEqual(r.status_code, 200)
        self.assertIn("models", r.json())

    def test_runs_validation(self):
        from fastapi.testclient import TestClient
        from app.main import app
        c = TestClient(app)
        r = c.post("/api/runs?provider=x&model=", files={"file": ("a.csv", b"a,b")})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
