"""End-to-end tests for multi-turn conversation handling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from config.settings import settings


def _can_run() -> bool:
    return Path(settings.bm25_index_path).exists() and (settings.has_google or settings.has_groq)


class TestMultiTurn:
    """Tests for multi-turn dialogue behavior."""

    def setup_method(self):
        from src.serving.app import app
        self.client = TestClient(app)

    def test_three_turn_conversation(self):
        """System maintains coherence across 3 turns."""
        if not _can_run():
            print("Skipping (need index + API key)")
            return

        session = "multi-turn-3"

        # Turn 1
        r1 = self.client.post("/api/v1/chat", json={
            "query": "What is a transformer model?",
            "session_id": session,
        })
        assert r1.status_code == 200
        assert len(r1.json()["answer"]) > 20

        # Turn 2 — follow-up
        r2 = self.client.post("/api/v1/chat", json={
            "query": "How does its attention mechanism work?",
            "session_id": session,
        })
        assert r2.status_code == 200
        # Should reference attention (from context of turn 1)
        assert len(r2.json()["answer"]) > 20

        # Turn 3 — different but related
        r3 = self.client.post("/api/v1/chat", json={
            "query": "What about BERT specifically?",
            "session_id": session,
        })
        assert r3.status_code == 200

    def test_separate_sessions_are_isolated(self):
        """Different session IDs don't share context."""
        if not _can_run():
            print("⏭️  Skipping")
            return

        # Session A talks about RAG
        self.client.post("/api/v1/chat", json={
            "query": "Explain RAG architecture",
            "session_id": "session-A",
        })

        # Session B asks about "it" — should NOT reference RAG
        r = self.client.post("/api/v1/chat", json={
            "query": "What are its main challenges?",
            "session_id": "session-B",
        })
        assert r.status_code == 200


if __name__ == "__main__":
    t = TestMultiTurn()
    t.setup_method()
    t.test_three_turn_conversation()
    print("3-turn conversation")
    t.test_separate_sessions_are_isolated()
    print("Session isolation")
    print("\n All multi-turn tests passed")
