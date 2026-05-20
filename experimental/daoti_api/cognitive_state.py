from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class CognitiveState:
    active_goals: list = field(default_factory=list)
    task_stack: list = field(default_factory=list)
    emotional_weights: dict = field(default_factory=dict)
    attention_focus: list = field(default_factory=list)
    belief_state: dict = field(default_factory=dict)
    uncertainty_map: dict = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def _touch(self):
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "active_goals": self.active_goals,
            "task_stack": self.task_stack,
            "emotional_weights": self.emotional_weights,
            "attention_focus": self.attention_focus,
            "belief_state": self.belief_state,
            "uncertainty_map": self.uncertainty_map,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveState":
        return cls(
            active_goals=d.get("active_goals", []),
            task_stack=d.get("task_stack", []),
            emotional_weights=d.get("emotional_weights", {}),
            attention_focus=d.get("attention_focus", []),
            belief_state=d.get("belief_state", {}),
            uncertainty_map=d.get("uncertainty_map", {}),
            session_id=d.get("session_id", str(uuid.uuid4())[:8]),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    def add_goal(self, goal: dict):
        goal.setdefault("id", str(uuid.uuid4())[:8])
        goal.setdefault("status", "active")
        goal.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.active_goals.append(goal)
        self._touch()

    def complete_goal(self, goal_id: str):
        for g in self.active_goals:
            if g.get("id") == goal_id:
                g["status"] = "completed"
                g["completed_at"] = datetime.now(timezone.utc).isoformat()
                break
        self._touch()

    def push_task(self, task: dict):
        task.setdefault("id", str(uuid.uuid4())[:8])
        self.task_stack.append(task)
        self._touch()

    def pop_task(self) -> Optional[dict]:
        if self.task_stack:
            self._touch()
            return self.task_stack.pop()
        return None

    def set_emotion(self, emotion: str, weight: float):
        self.emotional_weights[emotion] = max(0.0, min(1.0, weight))
        self._touch()

    def update_uncertainty(self, domain: str, level: float):
        self.uncertainty_map[domain] = max(0.0, min(1.0, level))
        self._touch()

    def set_focus(self, items: list):
        self.attention_focus = items
        self._touch()

    def update_belief(self, key: str, value):
        self.belief_state[key] = value
        self._touch()

    def reset(self):
        self.active_goals = []
        self.task_stack = []
        self.emotional_weights = {}
        self.attention_focus = []
        self.belief_state = {}
        self.uncertainty_map = {}
        self._touch()
