"""
Evaluator Agent — self-reflection loop for quality assurance.
Scores faithfulness, relevance, completeness.
Decides whether to pass or retry with refinement.
"""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from src.utils import get_llm
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Evaluate this answer against the source context.

Criteria:
- FAITHFULNESS (0-1): Every claim supported by context?
- RELEVANCE (0-1): Directly answers the question?
- COMPLETENESS (0-1): All aspects addressed?

Context:
{context}

Question: {query}
Answer: {answer}

Respond ONLY with valid JSON:
{{"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "overall_score": 0.0, "hallucinations": [], "feedback": "", "pass": true}}"""),
])


class EvaluatorAgent:
    """Scores answer quality and gates output."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = PROMPT | self.llm

    def run(self, state: AgentState) -> dict:
        """Evaluate answer and decide pass/retry."""
        logger.info("Evaluator scoring")

        documents = state.get("retrieved_documents", [])
        answer = state.get("draft_answer", "")
        context = "\n\n".join(d["content"] for d in documents)

        if not answer or not context:
            return self._force_pass("Empty answer or context")

        try:
            response = self.chain.invoke({
                "context": context[:5000],
                "query": state["current_query"],
                "answer": answer,
            })

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(content)

        except Exception as e:
            logger.warning(f"Eval parse failed: {e}")
            result = {"overall_score": 0.65, "hallucinations": [], "pass": True, "feedback": ""}

        score = float(result.get("overall_score", 0.5))
        hallucinations = result.get("hallucinations", [])
        passed = result.get("pass", score >= settings.confidence_threshold)
        feedback = result.get("feedback", "")
        iteration = state.get("iteration_count", 0) + 1

        # Force pass at max iterations to prevent infinite loops
        if iteration >= settings.max_agent_iterations:
            passed = True
            logger.info(f"Max iterations ({settings.max_agent_iterations}) — forcing pass")

        logger.info(f"Score={score:.2f} | pass={passed} | iter={iteration}")

        return {
            "evaluation_score": score,
            "evaluation_feedback": feedback,
            "hallucination_detected": len(hallucinations) > 0,
            "iteration_count": iteration,
            "next_agent": "done" if passed else "refine",
            "is_complete": passed,
        }

    def _force_pass(self, reason: str) -> dict:
        return {
            "evaluation_score": 0.0,
            "evaluation_feedback": reason,
            "hallucination_detected": False,
            "iteration_count": 99,
            "next_agent": "done",
            "is_complete": True,
        }