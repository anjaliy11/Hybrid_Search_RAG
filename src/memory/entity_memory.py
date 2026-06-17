"""
Entity Memory — tracks entities mentioned across conversation turns.
Helps with coreference resolution ("it", "that system", etc.).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    context: str  # Last context where entity was mentioned
    turn_index: int  # When it was last mentioned
    mentions: int = 1


class EntityMemory:
    """Tracks entities across conversation for coreference resolution."""

    def __init__(self, max_entities: int = 50):
        self.entities: Dict[str, Entity] = {}
        self.max_entities = max_entities
        self._turn_counter = 0

    def add_entities(self, names: List[str], context: str):
        """Record entities from a turn."""
        self._turn_counter += 1

        for name in names:
            key = name.lower().strip()
            if key in self.entities:
                self.entities[key].mentions += 1
                self.entities[key].context = context[:200]
                self.entities[key].turn_index = self._turn_counter
            else:
                self.entities[key] = Entity(
                    name=name,
                    context=context[:200],
                    turn_index=self._turn_counter,
                )

        # Evict old entities if over limit
        if len(self.entities) > self.max_entities:
            sorted_entities = sorted(
                self.entities.items(), key=lambda x: x[1].turn_index
            )
            for key, _ in sorted_entities[:len(self.entities) - self.max_entities]:
                del self.entities[key]

    def get_recent(self, n: int = 10) -> List[Entity]:
        """Get most recently mentioned entities."""
        sorted_e = sorted(self.entities.values(), key=lambda e: e.turn_index, reverse=True)
        return sorted_e[:n]

    def resolve(self, pronoun_context: str) -> Optional[str]:
        """Simple pronoun resolution based on recency."""
        recent = self.get_recent(3)
        if recent:
            return recent[0].name
        return None
