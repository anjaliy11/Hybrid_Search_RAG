"""Chat endpoint — multi-turn dialogue with agentic RAG."""

import logging
from fastapi import APIRouter, HTTPException

from src.agents.graph import agent_executor
from src.agents.voice_agent import VoiceAgent
from src.memory.conversation import ConversationMemory
from src.serving.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Session store (use Redis in production)
_sessions: dict = {}
_voice_agent = None


def _get_session(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def _get_voice_agent() -> VoiceAgent:
    global _voice_agent
    if _voice_agent is None:
        _voice_agent = VoiceAgent()
    return _voice_agent


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Multi-turn agentic RAG endpoint."""
    memory = _get_session(request.session_id)

    state = {
        "messages": memory.get_messages(),
        "current_query": request.query,
        "original_query": request.query,
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

    config = {"configurable": {"thread_id": request.session_id}}

    try:
        result = agent_executor.invoke(state, config)
        answer = result.get("draft_answer", "Unable to generate an answer.")

        # Voice mode formatting
        if request.voice_mode:
            voice = _get_voice_agent()
            answer = voice.format_for_voice(answer)

        memory.add_turn(request.query, answer, result.get("evaluation_score", 0.0))

        return ChatResponse(
            answer=answer,
            sources=result.get("cited_sources", []),
            confidence=result.get("evaluation_score", 0.0),
            hallucination_detected=result.get("hallucination_detected", False),
            iterations=result.get("iteration_count", 0),
            session_id=request.session_id,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))