"""
Hallucination Detector — claim-level verification.
Extracts atomic claims from answers and verifies each against source context.
"""

import json
import logging
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Verify the answer against the context for hallucinations.

For each factual claim, determine if it is supported by the context.

Context:
{context}

Answer:
{answer}

Respond ONLY with valid JSON:
{{
  "claims": [
    {{"text": "claim", "supported": true, "evidence": "quote or empty"}},
  ],
  "hallucination_rate": 0.0,
  "summary": "brief findings"
}}"""),
])


class HallucinationDetector:
    """Detects hallucinated claims via context verification."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = PROMPT | self.llm

    def detect(self, answer: str, context: str) -> Dict:
        """
        Run claim-level hallucination detection.
        
        Returns:
            {"hallucination_rate": float, "total_claims": int, 
             "unsupported_claims": list, "details": list}
        """
        if not answer.strip() or not context.strip():
            return {"hallucination_rate": 0.0, "total_claims": 0,
                    "unsupported_claims": [], "details": []}

        try:
            response = self.chain.invoke({
                "context": context[:4000],
                "answer": answer[:2000],
            })

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(content)
            claims = result.get("claims", [])

            unsupported = [c["text"] for c in claims if not c.get("supported", True)]
            total = max(len(claims), 1)

            return {
                "hallucination_rate": len(unsupported) / total,
                "total_claims": len(claims),
                "unsupported_claims": unsupported,
                "details": claims,
                "summary": result.get("summary", ""),
            }

        except Exception as e:
            logger.warning(f"Hallucination detection error: {e}")
            return {"hallucination_rate": 0.0, "total_claims": 0,
                    "unsupported_claims": [], "details": [], "error": str(e)}