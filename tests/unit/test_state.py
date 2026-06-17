"""Unit tests for agent state schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.state import AgentState


class TestAgentState:
    """Tests for state schema integrity."""

    def test_state_has_required_fields(self):
        """All required fields exist in the TypedDict."""
        required_fields = [
            "messages", "current_query", "original_query",
            "sub_queries", "research_plan",
            "retrieved_documents", "retrieval_attempts",
            "draft_answer", "cited_sources",
            "evaluation_score", "evaluation_feedback", "hallucination_detected",
            "next_agent", "iteration_count", "is_complete", "error",
        ]

        annotations = AgentState.__annotations__
        for field in required_fields:
            assert field in annotations, f"Missing field: {field}"

    def test_state_can_be_instantiated(self):
        """State dict can be created with all fields."""
        state: AgentState = {
            "messages": [],
            "current_query": "test query",
            "original_query": "test query",
            "sub_queries": [],
            "research_plan": "",
            "retrieved_documents": [],
            "retrieval_attempts": 0,
            "draft_answer": "",
            "cited_sources": [],
            "evaluation_score": 0.0,
            "evaluation_feedback": "",
            "hallucination_detected": False,
            "next_agent": "researcher",
            "iteration_count": 0,
            "is_complete": False,
            "error": None,
        }

        assert state["current_query"] == "test query"
        assert state["iteration_count"] == 0
        assert state["is_complete"] is False


if __name__ == "__main__":
    t = TestAgentState()
    t.test_state_has_required_fields()
    t.test_state_can_be_instantiated()
    print("All state tests passed")