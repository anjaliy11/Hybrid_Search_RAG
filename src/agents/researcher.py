"""
Researcher Agent — decomposes complex queries into searchable sub-questions.
Enables multi-hop reasoning by breaking queries into atomic retrievals.
"""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Decompose this query into searchable sub-questions.

Rules:
1. Simple/factual → return query as-is in sub_queries list
2. Complex/multi-hop → break into 2-4 independent sub-questions
3. Each sub-question must be self-contained (searchable alone)
4. Order by logical dependency

Query: {query}
Prior context: {context}

Respond ONLY with valid JSON (no markdown fences):
{{"is_complex": true, "sub_queries": ["q1", "q2"], "plan": "how to combine sub-answers"}}"""),
])


class ResearcherAgent:
    """Query decomposition for multi-hop reasoning."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = PROMPT | self.llm

    def run(self, state: AgentState) -> dict:
        """Decompose query into sub-questions."""
        query = state["current_query"]
        logger.info(f"Researcher: {query[:60]}...")

        context = self._get_context(state)

        try:
            response = self.chain.invoke({"query": query, "context": context})
            content = self._clean(response.content)
            result = json.loads(content)

            sub_queries = result.get("sub_queries", [query])
            plan = result.get("plan", "Direct answer")

            # Validate sub_queries
            if not sub_queries or not isinstance(sub_queries, list):
                sub_queries = [query]

        except Exception as e:
            logger.warning(f"Decomposition failed ({e}), using original query")
            sub_queries = [query]
            plan = "Direct retrieval"

        logger.info(f"Decomposed → {len(sub_queries)} sub-queries")
        return {
            "sub_queries": sub_queries,
            "research_plan": plan,
            "next_agent": "retriever",
        }

    def _get_context(self, state: AgentState) -> str:
        """Recent conversation for coreference resolution."""
        messages = state.get("messages", [])
        if not messages:
            return "None"
        recent = messages[-4:]
        return "\n".join(
            f"{getattr(m, 'type', '?')}: {getattr(m, 'content', '')[:120]}"
            for m in recent
        )

    def _clean(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return text