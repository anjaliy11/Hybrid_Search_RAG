"""
Benchmark Suite — end-to-end evaluation against golden QA sets.
"""

import json
import logging
import time
from typing import List, Dict
from pathlib import Path

from src.agents.graph import agent_executor
from src.evaluation.hallucination import HallucinationDetector

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs end-to-end benchmarks on eval sets."""

    def __init__(self):
        self.detector = HallucinationDetector()

    def run(self, eval_path: str = "./data/eval_sets/eval_set.json") -> Dict:
        """
        Execute benchmark on a QA eval set.
        
        Returns summary metrics and per-question details.
        """
        path = Path(eval_path)
        if not path.exists():
            raise FileNotFoundError(f"Eval set not found: {eval_path}")

        with open(path) as f:
            eval_set = json.load(f)

        results = []
        total_time = 0

        for i, item in enumerate(eval_set):
            start = time.time()

            state = self._build_state(item["question"])
            config = {"configurable": {"thread_id": f"bench-{i}"}}

            try:
                result = agent_executor.invoke(state, config)
                elapsed = time.time() - start
                total_time += elapsed

                context = "\n".join(d["content"] for d in result.get("retrieved_documents", []))
                hall = self.detector.detect(result.get("draft_answer", ""), context)

                results.append({
                    "question": item["question"],
                    "type": item.get("type", "unknown"),
                    "score": result.get("evaluation_score", 0.0),
                    "hallucination_rate": hall.get("hallucination_rate", 0.0),
                    "iterations": result.get("iteration_count", 0),
                    "latency_s": round(elapsed, 2),
                })

            except Exception as e:
                results.append({
                    "question": item["question"],
                    "error": str(e),
                    "score": 0.0,
                })

        return self._summarize(results, total_time)

    def _build_state(self, query: str) -> dict:
        return {
            "messages": [],
            "current_query": query,
            "original_query": query,
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

    def _summarize(self, results: List[Dict], total_time: float) -> Dict:
        valid = [r for r in results if "error" not in r]
        n = len(valid) or 1

        return {
            "total_questions": len(results),
            "successful": len(valid),
            "avg_score": sum(r["score"] for r in valid) / n,
            "avg_hallucination_rate": sum(r.get("hallucination_rate", 0) for r in valid) / n,
            "avg_iterations": sum(r.get("iterations", 0) for r in valid) / n,
            "avg_latency_s": total_time / len(results),
            "total_time_s": round(total_time, 1),
            "details": results,
        }