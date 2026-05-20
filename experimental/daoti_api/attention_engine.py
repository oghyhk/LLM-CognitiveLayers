import logging
from datetime import datetime, timezone

logger = logging.getLogger("daoti.attention")


class AttentionEngine:
    def __init__(self, decay_factor: float = 0.01):
        self.decay_factor = decay_factor

    def select_top_k(self, concept_graph, cognitive_state, context: str = "",
                     k: int = 5) -> list:
        scored = []
        for node_name in concept_graph.graph.nodes():
            node_data = concept_graph.graph.nodes[node_name]
            score = self._focus_score(
                node_name, node_data, context, cognitive_state
            )
            scored.append((node_name, node_data, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

    def _focus_score(self, node_name: str, node_data: dict, context: str,
                     cognitive_state) -> float:
        relevance = self._compute_relevance(node_name, node_data, context)
        recency = self._compute_recency(node_data)
        goal_alignment = self._compute_goal_alignment(
            node_name, cognitive_state.active_goals
        )
        emotional_weight = self._compute_emotional_weight(
            node_name, cognitive_state.emotional_weights
        )
        score = (
            relevance + recency + goal_alignment + emotional_weight
            - self.decay_factor
        )
        return max(0.0, min(1.0, score))

    def _compute_relevance(self, node_name: str, node_data: dict,
                           context: str) -> float:
        if not context:
            return 0.5
        name_lower = node_name.lower()
        ctx_lower = context.lower()
        words = name_lower.replace("_", " ").split()
        matches = sum(1 for w in words if w in ctx_lower)
        if matches > 0:
            return min(1.0, 0.3 + 0.3 * matches)
        return 0.1

    def _compute_recency(self, node_data: dict) -> float:
        ts_str = node_data.get("timestamp", "")
        if not ts_str:
            return 0.3
        try:
            ts = datetime.fromisoformat(ts_str)
            now = datetime.now(timezone.utc)
            delta = (now - ts).total_seconds()
            hours = delta / 3600
            if hours < 1:
                return 1.0
            if hours < 24:
                return 0.7
            if hours < 168:
                return 0.4
            return 0.1
        except (ValueError, TypeError):
            return 0.3

    def _compute_goal_alignment(self, node_name: str,
                                active_goals: list) -> float:
        if not active_goals:
            return 0.3
        name_lower = node_name.lower().replace("_", " ")
        total = 0.0
        for goal in active_goals:
            desc = goal.get("description", "").lower()
            if any(w in desc for w in name_lower.split()):
                total += 1.0
            elif name_lower in desc:
                total += 0.5
        return min(1.0, total / max(1, len(active_goals)))

    def _compute_emotional_weight(self, node_name: str,
                                  emotional_weights: dict) -> float:
        if not emotional_weights:
            return 0.3
        name_lower = node_name.lower()
        total = 0.0
        for emotion, weight in emotional_weights.items():
            if emotion.lower() in name_lower:
                total += weight
        return min(1.0, total)
