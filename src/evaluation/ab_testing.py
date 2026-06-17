"""
A/B Testing — compare prompt variants systematically.
"""

import time
import logging
from typing import Dict, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt version to test."""
    name: str
    version: str
    config: dict = field(default_factory=dict)
    scores: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / max(len(self.scores), 1)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / max(len(self.latencies), 1)


class ABTestRunner:
    """Compare prompt/config variants on same eval set."""

    def __init__(self):
        self.variants: Dict[str, PromptVariant] = {}

    def add_variant(self, name: str, version: str, config: dict = None):
        self.variants[name] = PromptVariant(name=name, version=version, config=config or {})

    def run(
        self,
        eval_questions: List[str],
        run_fn: Callable,  # (query, config) -> {"score": float}
    ) -> Dict:
        """
        Run all variants against questions.
        
        run_fn should accept (query: str, variant_config: dict) and return {"score": float}
        """
        for name, variant in self.variants.items():
            logger.info(f"Testing variant: {name}")

            for question in eval_questions:
                start = time.time()
                result = run_fn(question, variant.config)
                elapsed = time.time() - start

                variant.scores.append(result.get("score", 0.0))
                variant.latencies.append(elapsed)

        # Determine winner
        comparison = {}
        for name, v in self.variants.items():
            comparison[name] = {
                "avg_score": round(v.avg_score, 3),
                "avg_latency": round(v.avg_latency, 2),
                "n_samples": len(v.scores),
            }

        winner = max(self.variants.items(), key=lambda x: x[1].avg_score)
        comparison["winner"] = winner[0]

        return comparison
