import json
import logging
from typing import Optional

logger = logging.getLogger("daoti.planning")

PLAN_SCHEMA = {
    "plan": [
        {
            "step": "string",
            "reasoning": "string",
            "expected_outcome": "string",
            "dependencies": ["string"],
            "confidence": 0.8
        }
    ],
    "overall_confidence": 0.7,
    "risks": ["string"],
    "fallback": "string"
}


class PlanningEngine:
    def __init__(self, max_branches: int = 3, max_depth: int = 3):
        self.max_branches = max_branches
        self.max_depth = max_depth

    def plan(self, task_description: str, cognitive_state,
             concept_graph, memory_system, api_client) -> dict:
        if not self._is_complex(task_description):
            return {
                "plan": [],
                "reasoning": "Simple task, no planning needed",
                "response_mode": "direct"
            }

        goals = [g.get("description", "") for g in cognitive_state.active_goals]
        relevant_memories = memory_system.retrieve_all_recent(5)
        active_concepts = concept_graph.get_active_nodes(threshold=0.3)

        messages = [{"role": "user", "content": (
            f"Task: {task_description}\n"
            f"Active goals: {json.dumps(goals)}\n"
            f"Relevant context: {json.dumps([n[0] for n in active_concepts[:5]])}\n"
        )}]

        try:
            result = api_client.chat_structured(
                messages,
                PLAN_SCHEMA,
                "You are a planning engine. Decompose the task into structured steps."
            )
            return result
        except Exception as e:
            logger.warning(f"Planning failed, using direct mode: {e}")
            return {
                "plan": [],
                "reasoning": "Planning unavailable, answering directly",
                "response_mode": "direct"
            }

    def _is_complex(self, task: str) -> bool:
        complexity_markers = [
            "plan", "build", "create", "design", "implement",
            "analyze", "compare", "evaluate", "how to",
            "step by step", "what if", "explain why", "debug"
        ]
        task_lower = task.lower()
        score = sum(1 for m in complexity_markers if m in task_lower)
        return score >= 1 or len(task.split()) > 20

    def evaluate_plan(self, plan: dict, goals: list) -> float:
        if not plan.get("plan"):
            return 0.5
        conf = plan.get("overall_confidence", 0.5)
        steps = len(plan["plan"])
        step_bonus = min(0.2, steps * 0.05)
        return min(1.0, conf + step_bonus)

    def decompose_task(self, task: str) -> list:
        words = task.split()
        if len(words) <= 15:
            return [task]
        mid = len(words) // 2
        return [" ".join(words[:mid]), " ".join(words[mid:])]
