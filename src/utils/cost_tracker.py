"""Token usage tracking across LLM calls."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class UsageStats:
    """Tracks cumulative token usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_model: Dict[str, dict] = field(default_factory=dict)

    def record(self, model: str, input_tok: int, output_tok: int):
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.calls += 1

        if model not in self.by_model:
            self.by_model[model] = {"input": 0, "output": 0, "calls": 0}
        self.by_model[model]["input"] += input_tok
        self.by_model[model]["output"] += output_tok
        self.by_model[model]["calls"] += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return f"Tokens: {self.total_tokens:,} (in:{self.input_tokens:,} out:{self.output_tokens:,}) | Calls: {self.calls}"


# Global singleton
usage = UsageStats()