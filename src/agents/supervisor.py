"""
Supervisor Agent — routes queries based on complexity analysis.
Decides whether a query needs decomposition or can go directly to retrieval.
"""

import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Classify this query's complexity for routing:

Query: {query}

Indicators of complexity requiring decomposition:
- Compares multiple concepts
- Requires temporal reasoning
- Asks "how X relates to Y"
- Contains multiple sub-questions
- Needs information from different topics

Simple queries (direct retrieval):
- Single factual question
- Definition request
- Single concept explanation

Respond ONLY with JSON: {{"route": "researcher" or "retriever", "reason": "brief"}}"""),
])


class SupervisorAgent:
    """Routes queries to appropriate first agent based on complexity."""

    def __init__(self):
        self.llm = get_fast_llm()
        self.chain = ROUTE_PROMPT | self.llm

    def run(self, state: AgentState) -> dict:
        """Analyze query and decide routing."""
        query = state["current_query"]

        try:
            response = self.chain.invoke({"query": query})
            content = self._clean_json(response.content)
            result = json.loads(content)
            route = result.get("route", "researcher")
        except Exception as e:
            logger.debug(f"Supervisor routing failed: {e}, defaulting to researcher")
            route = "researcher"

        logger.info(f"Supervisor routed to: {route}")
        return {"next_agent": route}

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return text