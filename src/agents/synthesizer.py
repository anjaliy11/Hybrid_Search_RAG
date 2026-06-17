"""
Synthesizer Agent — generates grounded, cited answers from context.
Enforces anti-hallucination constraints via prompt design.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm
from src.agents.state import AgentState
from src.retrieval.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Generate an accurate answer based ONLY on the context below.

RULES:
1. ONLY use information from the context documents
2. Cite every claim with [Doc N] notation
3. If context is insufficient, say "Based on available documents, I cannot fully answer this"
4. NEVER fabricate facts, numbers, or claims not in context
5. Lead with the key insight, then supporting detail
6. Be concise and well-structured

Context:
{context}

Research Plan: {plan}
Question: {query}

Grounded answer:"""),
])


class SynthesizerAgent:
    """Generates cited, grounded answers from retrieved context."""

    def __init__(self):
        self.llm = get_llm(temperature=0.15)
        self.chain = PROMPT | self.llm
        self.context_builder = ContextBuilder()

    def run(self, state: AgentState) -> dict:
        """Generate answer grounded in retrieved documents."""
        logger.info("Synthesizer generating answer")

        documents = state.get("retrieved_documents", [])
        if not documents:
            return {
                "draft_answer": "No relevant documents found to answer this question.",
                "cited_sources": [],
                "next_agent": "evaluator",
            }

        context = self.context_builder.build_from_dicts(documents)
        plan = state.get("research_plan", "Direct answer")

        try:
            response = self.chain.invoke({
                "context": context,
                "plan": plan,
                "query": state["current_query"],
            })
            answer = response.content
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            answer = f"Generation failed: {str(e)}"

        cited = self._extract_citations(answer, documents)

        return {
            "draft_answer": answer,
            "cited_sources": cited,
            "next_agent": "evaluator",
        }

    def _extract_citations(self, answer: str, documents: list) -> list:
        """Identify which documents were cited in the answer."""
        cited = []
        for i, doc in enumerate(documents):
            if f"[Doc {i+1}]" in answer:
                source = doc.get("metadata", {}).get("source", f"doc_{i+1}")
                cited.append(source)
        return cited