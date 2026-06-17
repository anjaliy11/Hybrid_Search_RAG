"""
Agent State Schema — the shared data structure flowing through the graph.
Every agent reads from and writes to specific fields of this state.
"""

from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state for the multi-agent workflow.
    
    Field groups:
      - Conversation: message history and current query
      - Research: decomposed sub-queries and plan
      - Retrieval: fetched documents and attempt count
      - Synthesis: generated answer and citations
      - Evaluation: quality scores and hallucination flags
      - Control: routing decisions and iteration tracking
    """
    # Conversation
    messages: Annotated[list, add_messages]
    current_query: str
    original_query: str

    # Research
    sub_queries: List[str]
    research_plan: str

    # Retrieval
    retrieved_documents: List[dict]
    retrieval_attempts: int

    # Synthesis
    draft_answer: str
    cited_sources: List[str]

    # Evaluation
    evaluation_score: float
    evaluation_feedback: str
    hallucination_detected: bool

    # Control
    next_agent: str
    iteration_count: int
    is_complete: bool
    error: Optional[str]