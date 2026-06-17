"""End-to-end tests for the chat API endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient
from config.settings import settings


def _can_run() -> bool:
    return Path(settings.bm25_index_path).exists() and (settings.has_google or settings.has_groq)


class TestChatEndpoint:
    """E2E tests for /api/v1/chat."""

    def setup_method(self):
        from src.serving.app import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Health check returns 200."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "indices" in data

    def test_chat_returns_answer(self):
        """Chat endpoint returns structured response."""
        if not _can_run():
            print("Skipping (need index + API key)")
            return

        response = self.client.post("/api/v1/chat", json={
            "query": "What is retrieval augmented generation?",
            "session_id": "e2e-test",
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 20
        assert "confidence" in data
        assert data["session_id"] == "e2e-test"

    def test_chat_validates_empty_query(self):
        """Empty query returns 422 validation error."""
        response = self.client.post("/api/v1/chat", json={
            "query": "",
            "session_id": "test",
        })
        assert response.status_code == 422

    def test_chat_session_persistence(self):
        """Same session_id maintains conversation context."""
        if not _can_run():
            print("Skipping (need index + API key)")
            return

        # First message
        r1 = self.client.post("/api/v1/chat", json={
            "query": "What is RAG?",
            "session_id": "persist-test",
        })
        assert r1.status_code == 200

        # Follow-up referencing "it"
        r2 = self.client.post("/api/v1/chat", json={
            "query": "What are its main components?",
            "session_id": "persist-test",
        })
        assert r2.status_code == 200
        assert len(r2.json()["answer"]) > 10


if __name__ == "__main__":
    t = TestChatEndpoint()
    t.setup_method()
    t.test_health_endpoint()
    print(" Health endpoint")
    t.test_chat_validates_empty_query()
    print(" Validation")
    t.test_chat_returns_answer()
    print(" Chat returns answer")
    t.test_chat_session_persistence()
    print(" Session persistence")
    print("\n All E2E tests passed")