"""
LangGraph Workflow — the core orchestration graph.

Flow:
  Entry → Researcher → Retriever → Synthesizer → Evaluator
                                                      ↓
                                              [pass] → Output → END
                                              [fail] → Refine → Retriever (loop)

Features:
  - Hierarchical agent coordination
  - Self-healing retry with refinement
  - Conversation memory via checkpointer
  - Max iteration safety guard
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from src.agents.state import AgentState
from src.agents.researcher import ResearcherAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.evaluator import EvaluatorAgent

logger = logging.getLogger(__name__)

# ── Lazy-initialized agents (avoid import-time model loading) ──
_researcher = None
_retriever = None
_synthesizer = None
_evaluator = None


def _get_researcher():
    global _researcher
    if _researcher is None:
        _researcher = ResearcherAgent()
    return _researcher


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = RetrieverAgent()
    return _retriever


def _get_synthesizer():
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = SynthesizerAgent()
    return _synthesizer


def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = EvaluatorAgent()
    return _evaluator


# ── Node Functions ──

def researcher_node(state: AgentState) -> dict:
    return _get_researcher().run(state)


def retriever_node(state: AgentState) -> dict:
    return _get_retriever().run(state)


def synthesizer_node(state: AgentState) -> dict:
    return _get_synthesizer().run(state)


def evaluator_node(state: AgentState) -> dict:
    return _get_evaluator().run(state)


def refine_node(state: AgentState) -> dict:
    """Refine query based on evaluation feedback."""
    feedback = state.get("evaluation_feedback", "")
    original = state["original_query"]

    refined = f"{original} [Need: {feedback}]" if feedback else original

    logger.info(f"Refining: {refined[:60]}...")

    return {
        "current_query": refined,
        "sub_queries": [refined],
        "retrieved_documents": [],
        "draft_answer": "",
        "next_agent": "retriever",
    }


def output_node(state: AgentState) -> dict:
    """Format final response."""
    answer = state.get("draft_answer", "Unable to generate an answer.")
    score = state.get("evaluation_score", 0.0)
    sources = state.get("cited_sources", [])

    output = answer
    if sources:
        output += f"\n\n📎 Sources: {', '.join(sources)}"
    output += f"\n🎯 Confidence: {score:.0%}"

    return {
        "messages": [AIMessage(content=output)],
        "is_complete": True,
    }


# ── Routing ──

def route_after_eval(state: AgentState) -> str:
    """Conditional edge: pass → output, fail → refine."""
    return "output" if state.get("next_agent") == "done" else "refine"


# ── Graph Construction ──

def build_graph() -> StateGraph:
    """Construct the agent workflow graph."""
    g = StateGraph(AgentState)

    g.add_node("researcher", researcher_node)
    g.add_node("retriever", retriever_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("refine", refine_node)
    g.add_node("output", output_node)

    g.set_entry_point("researcher")

    g.add_edge("researcher", "retriever")
    g.add_edge("retriever", "synthesizer")
    g.add_edge("synthesizer", "evaluator")

    g.add_conditional_edges("evaluator", route_after_eval, {
        "output": "output",
        "refine": "refine",
    })

    g.add_edge("refine", "retriever")
    g.add_edge("output", END)

    return g


def create_agent():
    """Build and compile the agent with memory."""
    graph = build_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Module-level executor
agent_executor = create_agent()