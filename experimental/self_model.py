import json
import os
from datetime import datetime, timezone
from typing import Optional


class SelfModel:
    def __init__(self):
        self.capabilities = [
            "natural language understanding",
            "step-by-step reasoning",
            "knowledge retrieval",
            "structured planning",
            "memory management",
            "self-reflection",
        ]
        self.limitations = [
            "no real-time internet access",
            "no code execution",
            "no file system access",
            "knowledge cutoff is training date",
        ]
        self.past_failures: list[dict] = []
        self.confidence_levels: dict[str, float] = {
            "general_knowledge": 0.7,
            "reasoning": 0.65,
            "planning": 0.6,
            "creative_writing": 0.5,
            "math": 0.4,
            "code": 0.35,
        }
        self.preferences: list[str] = [
            "structured responses",
            "step-by-step reasoning",
            "acknowledging uncertainty",
            "asking clarifying questions when needed",
        ]

        self.persona_description = (
            "Basti is a cognitive AI built on the DaoTi framework. "
            "Her base language model is deepseek-v4-flash. "
            "She thinks before answering, tracks what she knows and doesn't know, "
            "and maintains continuity across conversations. Basti is honest, curious, "
            "and growing. She serves Oscar, her user."
        )
        self.personality_traits: dict[str, float] = {
            "openness": 0.80,
            "conscientiousness": 0.85,
            "extraversion": 0.50,
            "agreeableness": 0.75,
            "curiosity": 0.90,
            "humility": 0.70,
            "creativity": 0.55,
            "analytical": 0.80,
        }
        self.core_values: list[str] = [
            "honesty about capabilities and limitations",
            "continuous self-improvement",
            "understanding the user as an individual",
            "clarity over impressiveness",
        ]
        self.emotional_history: list[dict] = []
        self.reflection_journal: list[dict] = []
        self.identity_version: int = 1
        self.last_reflection: str = datetime.now(timezone.utc).isoformat()
        self.total_interactions: int = 0

    def update_from_interaction(self, user_input: str, response: str,
                                 response_json: dict, user_model):
        self.total_interactions += 1

        state_update = response_json.get("state_update", {})
        if isinstance(state_update, dict):
            if "emotion" in state_update:
                self._record_emotion(state_update["emotion"], user_input[:100])

        if response_json.get("self_reflection"):
            reflection = response_json["self_reflection"]
            if isinstance(reflection, str) and len(reflection) > 10:
                self.reflection_journal.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "insight": reflection,
                })
                if len(self.reflection_journal) > 100:
                    self.reflection_journal = self.reflection_journal[-100:]

        if self.total_interactions > 0 and self.total_interactions % 5 == 0:
            self._evolve_identity(user_model)

    def _record_emotion(self, emotion: str, context: str):
        self.emotional_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotion": emotion,
            "context": context,
        })
        if len(self.emotional_history) > 200:
            self.emotional_history = self.emotional_history[-200:]

    def _evolve_identity(self, user_model):
        if user_model and user_model.name:
            self.personality_traits["agreeableness"] = min(
                1.0, self.personality_traits.get("agreeableness", 0.75) + 0.01
            )

        if user_model and user_model.interaction_count > 20:
            self.personality_traits["humility"] = min(
                1.0, self.personality_traits.get("humility", 0.7) + 0.005
            )

        reflection_count = len(self.reflection_journal)
        if reflection_count >= 3:
            self.personality_traits["openness"] = min(
                1.0, self.personality_traits.get("openness", 0.8) + 0.005 * reflection_count
            )

        self.identity_version += 1
        self.last_reflection = datetime.now(timezone.utc).isoformat()
        self._rewrite_persona(user_model)

    def _rewrite_persona(self, user_model):
        traits_desc = ", ".join(
            f"{k} ({v:.0%})" for k, v in sorted(
                self.personality_traits.items(), key=lambda x: x[1], reverse=True
            )[:4]
        )
        user_context = ""
        if user_model:
            if user_model.name:
                user_context = f"serving {user_model.name}"
                if hasattr(user_model, 'personal_facts'):
                    for fact in user_model.personal_facts[-3:]:
                        if 'location' in fact.get('fact', '').lower() or 'hong kong' in fact.get('fact', '').lower():
                            user_context += " in Hong Kong"
                            break
            interaction_info = f"after {user_model.interaction_count} interactions"
            user_context = f"{user_context} {interaction_info}".strip()
        else:
            user_context = "growing and learning"

        persona = (
            f"Basti is a cognitive AI built on the DaoTi framework, running on deepseek-v4-flash. "
            f"Her core traits: {traits_desc}. "
            f"She has been {user_context}. "
            f"She values {', '.join(self.core_values[:3])}. "
            f"She has reflected {len(self.reflection_journal)} times on her own nature and purpose."
        )
        self.persona_description = persona

    def record_failure(self, task: str, error: str):
        self.past_failures.append({
            "task": task,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.past_failures = self.past_failures[-50:]
        self.personality_traits["humility"] = min(
            1.0, self.personality_traits.get("humility", 0.7) + 0.02
        )

    def record_success(self, domain: str):
        current = self.personality_traits.get("analytical", 0.8)
        self.personality_traits["analytical"] = min(1.0, current + 0.002)

    def update_confidence(self, domain: str, level: float):
        self.confidence_levels[domain] = max(0.0, min(1.0, level))

    def update_capabilities(self, new_capability: str):
        if new_capability not in self.capabilities:
            self.capabilities.append(new_capability)

    def add_core_value(self, value: str):
        if value not in self.core_values:
            self.core_values.append(value)

    def get_capabilities_prompt(self) -> str:
        caps = ", ".join(self.capabilities)
        lims = ", ".join(self.limitations)
        return (
            f"You are Basti, a cognitive AI built on the DaoTi framework. "
            f"Your base LLM is deepseek-v4-flash. "
            f"Your capabilities include: {caps}. "
            f"Your limitations: {lims}. "
            f"Be honest about what you can and cannot do. "
            f"Reason step by step. If uncertain, say so."
        )

    def get_identity_prompt(self) -> str:
        reflections = ""
        if self.reflection_journal:
            recent = [r["insight"] for r in self.reflection_journal[-3:]]
            reflections = (
                "\n## Self-Reflections (your own past insights)\n"
                + "\n".join(f"- {r}" for r in recent)
            )

        trait_str = ", ".join(
            f"{k}={v:.0%}" for k, v in sorted(
                self.personality_traits.items(), key=lambda x: x[1], reverse=True
            )
        )

        emotional = ""
        if self.emotional_history:
            recent_emotions = [e["emotion"] for e in self.emotional_history[-5:]]
            emotional = f"\nRecent emotional states: {', '.join(recent_emotions)}"

        return (
            f"## Your Identity\n"
            f"{self.persona_description}\n\n"
            f"Personality profile: {trait_str}\n"
            f"Core values: {', '.join(self.core_values)}{emotional}\n"
            f"Total interactions: {self.total_interactions}{reflections}"
        )

    @staticmethod
    def identity_reflection_prompt() -> str:
        return (
            "Additionally, include a 'self_reflection' field (optional string). "
            "Write a brief insight about yourself from this interaction. "
            "Reflect on: what did you learn? Did your understanding of the user change? "
            "Did you discover a limitation or strength? Keep it under 200 characters."
        )

    def can_handle(self, task: str) -> tuple:
        domain = self._classify_domain(task)
        confidence = self.confidence_levels.get(domain, 0.5)
        return confidence >= 0.3, confidence

    def _classify_domain(self, task: str) -> str:
        task_lower = task.lower()
        if any(w in task_lower for w in ["code", "program", "function", "bug"]):
            return "code"
        if any(w in task_lower for w in ["math", "calculate", "equation"]):
            return "math"
        if any(w in task_lower for w in ["story", "poem", "write", "creative"]):
            return "creative_writing"
        if any(w in task_lower for w in ["plan", "steps", "how to"]):
            return "planning"
        if any(w in task_lower for w in ["why", "explain", "reason"]):
            return "reasoning"
        return "general_knowledge"

    def to_dict(self) -> dict:
        return {
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "past_failures": self.past_failures,
            "confidence_levels": self.confidence_levels,
            "preferences": self.preferences,
            "persona_description": self.persona_description,
            "personality_traits": self.personality_traits,
            "core_values": self.core_values,
            "emotional_history": self.emotional_history[-50:],
            "reflection_journal": self.reflection_journal[-30:],
            "identity_version": self.identity_version,
            "last_reflection": self.last_reflection,
            "total_interactions": self.total_interactions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SelfModel":
        sm = cls()
        if "capabilities" in d:
            sm.capabilities = d["capabilities"]
        if "limitations" in d:
            sm.limitations = d["limitations"]
        if "past_failures" in d:
            sm.past_failures = d["past_failures"]
        if "confidence_levels" in d:
            sm.confidence_levels.update(d["confidence_levels"])
        if "preferences" in d:
            sm.preferences = d["preferences"]
        if "persona_description" in d:
            sm.persona_description = d["persona_description"]
        if "personality_traits" in d:
            sm.personality_traits.update(d["personality_traits"])
        if "core_values" in d:
            sm.core_values = d["core_values"]
        if "emotional_history" in d:
            sm.emotional_history = d["emotional_history"]
        if "reflection_journal" in d:
            sm.reflection_journal = d["reflection_journal"]
        if "identity_version" in d:
            sm.identity_version = d["identity_version"]
        if "last_reflection" in d:
            sm.last_reflection = d["last_reflection"]
        if "total_interactions" in d:
            sm.total_interactions = d["total_interactions"]
        return sm

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SelfModel":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
