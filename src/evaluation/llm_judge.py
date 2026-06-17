"""
LLM-as-Judge — uses a separate LLM call to evaluate answer quality.
More nuanced than rule-based scoring.
"""

import json
import logging
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm

logger = logging.getLogger(__name__)

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """You are an expert evaluator. Rate this Q&A pair.

Question: {question}
Answer: {answer}
Reference (ground truth): {reference}

Rate on:
1. Correctness (0-1): factually accurate vs reference?
2. Clarity (0-1): well-structured and clear?
3. Conciseness (0-1): no unnecessary information?

Respond ONLY with JSON:
{{"correctness": 0.0, "clarity": 0.0, "conciseness": 0.0, "overall": 0.0, "reasoning": ""}}"""),
])


class LLMJudge:
    """LLM-as-Judge for answer quality evaluation."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = JUDGE_PROMPT | self.llm

    def judge(self, question: str, answer: str, reference: str = "") -> Dict:
        """
        Judge answer quality using LLM.
        
        Returns dict with correctness, clarity, conciseness, overall scores.
        """
        try:
            response = self.chain.invoke({
                "question": question,
                "answer": answer,
                "reference": reference or "Not provided",
            })

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            return json.loads(content)

        except Exception as e:
            logger.warning(f"LLM judge error: {e}")
            return {"correctness": 0.5, "clarity": 0.5, "conciseness": 0.5,
                    "overall": 0.5, "error": str(e)}