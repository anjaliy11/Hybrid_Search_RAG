"""
RAGAS Evaluation — industry-standard RAG metrics.
Graceful fallback if RAGAS has dependency issues.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """RAGAS metrics with fallback heuristic evaluation."""

    def __init__(self):
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            from ragas.metrics import faithfulness, answer_relevancy
            return True
        except (ImportError, Exception) as e:
            logger.info(f"RAGAS not available ({e}), using heuristic fallback")
            return False

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> Dict[str, float]:
        """Evaluate a single QA pair."""
        if self._available:
            return self._ragas_eval(question, answer, contexts, ground_truth)
        return self._heuristic_eval(question, answer, contexts)

    def _ragas_eval(
        self, question: str, answer: str, contexts: List[str], ground_truth: str
    ) -> Dict[str, float]:
        """Full RAGAS evaluation."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

            return {
                "faithfulness": float(results.get("faithfulness", 0)),
                "relevancy": float(results.get("answer_relevancy", 0)),
                "method": "ragas",
            }
        except Exception as e:
            logger.warning(f"RAGAS eval failed: {e}")
            return self._heuristic_eval(question, answer, contexts)

    def _heuristic_eval(
        self, question: str, answer: str, contexts: List[str]
    ) -> Dict[str, float]:
        """Keyword-overlap heuristic when RAGAS unavailable."""
        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())
        q_words = set(question.lower().split()) - {"what", "how", "why", "the", "is", "a", "an"}

        faithfulness = min(len(answer_words & context_words) / max(len(answer_words), 1) * 1.5, 1.0)
        relevancy = min(len(q_words & answer_words) / max(len(q_words), 1) * 1.5, 1.0)

        return {"faithfulness": faithfulness, "relevancy": relevancy, "method": "heuristic"}