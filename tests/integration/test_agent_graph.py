"""Integration test for the full agent graph execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings


def _has_prerequisites() -> bool:
    """Check if indices and API keys are available."""
    has_index = Path(settings.bm25_index_path).exists()
    has_llm = settings.has_google or settings.has_groq
    return has_index and has_llm


class TestAgentGraph:
    """Tests for the compiled LangGraph agent."""

    def test_agent_completes_simple_query(self):
        """Agent handles simple factual queries end-to-end."""
        if not _has_prerequisites():
            print(" Skipping (need index + API key)")
            return

        from src.agents.graph import agent_executor

        state = {
            "messages": [], "current_query": "What is RAG?",
            "original_query": "What is RAG?",
            "sub_queries": [], "research_plan": "",
            "retrieved_documents": [], "retrieval_attempts": 0,
            "draft_answer": "", "cited_sources": [],
            "evaluation_score": 0.0, "evaluation_feedback": "",
            "hallucination_detected": False, "next_agent": "researcher",
            "iteration_count": 0, "is_complete": False, "error": None,
        }

        config = {"configurable": {"thread_id": "test-simple"}}
        result = agent_executor.invoke(state, config)

        assert result["is_complete"] is True
        assert result["draft_answer"] != ""
        assert result["iteration_count"] <= settings.max_agent_iterations

    def test_agent_completes_complex_query(self):
        """Agent handles multi-hop queries."""
        if not _has_prerequisites():
            print(" Skipping (need index + API key)")
            return

        from src.agents.graph import agent_executor

        state = {
            "messages": [],
            "current_query": "How do transformers use attention differently from RNNs?",
            "original_query": "How do transformers use attention differently from RNNs?",
            "sub_queries": [], "research_plan": "",
            "retrieved_documents": [], "retrieval_attempts": 0,
            "draft_answer": "", "cited_sources": [],
            "evaluation_score": 0.0, "evaluation_feedback": "",
            "hallucination_detected": False, "next_agent": "researcher",
            "iteration_count": 0, "is_complete": False, "error": None,
        }

        config = {"configurable": {"thread_id": "test-complex"}}
        result = agent_executor.invoke(state, config)

        assert result["is_complete"] is True
        assert result["draft_answer"] != ""

    def test_agent_respects_max_iterations(self):
        """Agent doesn't loop forever."""
        if not _has_prerequisites():
            print(" Skipping (need index + API key)")
            return

        from src.agents.graph import agent_executor

        state = {
            "messages": [],
            "current_query": "Explain quantum consciousness in AI systems",  # unlikely to find good context
            "original_query": "Explain quantum consciousness in AI systems",
            "sub_queries": [], "research_plan": "",
            "retrieved_documents": [], "retrieval_attempts": 0,
            "draft_answer": "", "cited_sources": [],
            "evaluation_score": 0.0, "evaluation_feedback": "",
            "hallucination_detected": False, "next_agent": "researcher",
            "iteration_count": 0, "is_complete": False, "error": None,
        }

        config = {"configurable": {"thread_id": "test-maxiter"}}
        result = agent_executor.invoke(state, config)

        assert result["is_complete"] is True
        assert result["iteration_count"] <= settings.max_agent_iterations


if __name__ == "__main__":
    t = TestAgentGraph()
    t.test_agent_completes_simple_query()
    print("Simple query")
    t.test_agent_completes_complex_query()
    print(" Complex query")
    t.test_agent_respects_max_iterations()
    print(" Max iterations respected")
    print("\n All agent graph integration tests passed")