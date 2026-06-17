"""
Rubric-Based Scorer — evaluates answers against predefined rubrics.
Loads scoring criteria from YAML config files.
"""

import json
import logging
from typing import Dict
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_llm

logger = logging.getLogger(__name__)

RUBRIC_PROMPT = ChatPromptTemplate.from_messages([
    ("human", """Score this answer using the rubric below.

Rubric: {rubric_name}
Description: {rubric_description}
Scale: {scale}
Criteria:
{criteria}

Scoring Guide:
{scoring_guide}

---

Question: {question}
Context: {context}
Answer: {answer}

Respond ONLY with JSON:
{{"score": 0.0, "reasoning": "brief justification"}}"""),
])


class RubricScorer:
    """Evaluates answers against configurable YAML rubrics."""

    RUBRIC_DIR = Path("config/eval_rubrics")

    def __init__(self):
        self.llm = get_llm(temperature=0.0)
        self.chain = RUBRIC_PROMPT | self.llm
        self.rubrics = self._load_rubrics()

    def _load_rubrics(self) -> Dict:
        """Load all rubric YAML files."""
        rubrics = {}
        if not self.RUBRIC_DIR.exists():
            return rubrics

        for path in self.RUBRIC_DIR.glob("*.yaml"):
            try:
                with open(path) as f:
                    rubrics[path.stem] = yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load rubric {path.name}: {e}")

        logger.info(f"Loaded {len(rubrics)} evaluation rubrics")
        return rubrics

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        rubric_name: str = "faithfulness",
    ) -> Dict:
        """
        Score answer against a specific rubric.
        
        Returns: {"score": float, "reasoning": str, "rubric": str}
        """
        rubric = self.rubrics.get(rubric_name)
        if not rubric:
            return {"score": 0.5, "reasoning": f"Rubric '{rubric_name}' not found"}

        criteria = "\n".join(f"- {c}" for c in rubric.get("criteria", []))
        scoring = "\n".join(
            f"  {k}: {v}" for k, v in rubric.get("scoring", {}).items()
        )

        try:
            response = self.chain.invoke({
                "rubric_name": rubric.get("name", rubric_name),
                "rubric_description": rubric.get("description", ""),
                "scale": rubric.get("scale", "0.0 - 1.0"),
                "criteria": criteria,
                "scoring_guide": scoring,
                "question": question,
                "context": context[:3000],
                "answer": answer,
            })

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(content)
            result["rubric"] = rubric_name
            return result

        except Exception as e:
            logger.warning(f"Rubric scoring error: {e}")
            return {"score": 0.5, "reasoning": str(e), "rubric": rubric_name}

    def score_all_rubrics(self, question: str, answer: str, context: str) -> Dict:
        """Score against all available rubrics."""
        results = {}
        for name in self.rubrics:
            results[name] = self.score(question, answer, context, name)
        
        if results:
            avg = sum(r.get("score", 0) for r in results.values()) / len(results)
            results["overall"] = avg
        
        return results