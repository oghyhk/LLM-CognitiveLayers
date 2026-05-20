import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("daoti.local_runtime")

LOCAL_OUTPUT_SCHEMA = {
    "intent": "string",
    "state_update": {},
    "memory_write": "string or null",
    "reasoning": "string",
    "response": "string",
    "uncertainty": 0.5,
    "sentiment": 0.0,
    "user_facts": ["string"],
}

LOCAL_SCHEMA_DESC = '{"intent":"str","response":"str","reasoning":"str","uncertainty":0.0-1.0,"sentiment":-1.0to1.0,"user_facts":["str"]}'

LOCAL_SYSTEM_PROMPT = """You are Basti (local), a cognitive AI running on-device via Gemma 2B.

{identity_context}

User: {user_context}

State: {state_summary}

Concepts: {active_concepts}

Memories: {memory_summary}

Reply in JSON: {output_schema}

Instructions: reason first, then respond. Be honest about limits. Keep reasoning short (1-2 sentences).

Reply ONLY with valid JSON. No other text."""


class LocalCognitiveRuntime:
    def __init__(self, local_client, cognitive_state, concept_graph,
                 attention_engine, memory_system, planning_engine,
                 world_model, self_model, user_model=None):
        self.llm = local_client
        self.state = cognitive_state
        self.graph = concept_graph
        self.attention = attention_engine
        self.memory = memory_system
        self.planner = planning_engine
        self.world = world_model
        self.self_model = self_model
        self.user_model = user_model
        self.conversation_history = []
        self.current_step = "idle"
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._cached_identity = None
        self._cached_identity_version = 0

    def process(self, user_input: str) -> str:
        self.current_step = "parsing input"
        logger.info(f"Local: Processing: {user_input[:60]}...")

        self.current_step = "updating state"
        self.state.set_focus(self._extract_keywords(user_input))
        self.world.update_entity("user", {
            "last_input": user_input,
            "last_active": datetime.now(timezone.utc).isoformat()
        })

        self.current_step = "spreading activation"
        self.graph.spread_activation(self.state.attention_focus[:5], steps=2)
        active = self.attention.select_top_k(self.graph, self.state, user_input, k=4)
        active_names = [n[0] for n in active]

        self.current_step = "retrieving memories"
        memory_context = self.memory.get_memory_summary()

        self.current_step = "planning"
        plan_result = self.planner.plan(user_input, self.state, self.graph, self.memory, None)

        can_handle, confidence = self.self_model.can_handle(user_input)

        goals = [g.get("description", "") for g in self.state.active_goals]
        state_summary = (
            f"goals={goals}, confidence={confidence:.2f}"
            if goals else f"confidence={confidence:.2f}"
        )

        self.current_step = "building identity context"
        if self._cached_identity is None or self.self_model.identity_version != self._cached_identity_version:
            self._cached_identity = self._get_compact_identity()
            self._cached_identity_version = self.self_model.identity_version
        identity_context = self._cached_identity

        self.current_step = "loading user profile"
        user_context = self.user_model.get_user_context() if self.user_model else "No user profile."

        system_prompt = LOCAL_SYSTEM_PROMPT.format(
            identity_context=identity_context,
            user_context=user_context[:300],
            state_summary=state_summary,
            active_concepts=", ".join(active_names[:3]) if active_names else "none",
            memory_summary=memory_context[:300] if memory_context else "none",
            output_schema=LOCAL_SCHEMA_DESC,
        )

        if plan_result.get("response_mode") == "direct":
            system_prompt += "\nShort answer. Be concise."
        if can_handle:
            system_prompt += f"\nConfidence: {confidence:.2f}"
        else:
            system_prompt += f"\nLow confidence ({confidence:.2f}). Be careful."

        messages = [{"role": "user", "content": user_input}]

        self.current_step = "running local LLM"
        try:
            result = self.llm.chat_structured(messages, LOCAL_OUTPUT_SCHEMA, system_prompt)
        except Exception as e:
            self.current_step = "idle"
            logger.error(f"Local LLM error: {e}")
            return f"[Local error: {e}]"

        self.last_usage = result.pop("_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        if result.get("_raw"):
            response_text = result.get("response", str(result))
        else:
            response_text = result.get("response", str(result))
            if not response_text:
                response_text = result.get("reasoning", "I'm thinking...") or "Let me think about that."

        self._update_from_result(result, user_input, response_text)

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        self.current_step = "idle"
        return response_text

    def _update_from_result(self, result: dict, user_input: str, response_text: str):
        memory_text = result.get("memory_write")
        if memory_text:
            self.memory.store_episodic(
                f"User: {user_input[:100]} | Response: {memory_text[:100]}",
                {"type": "interaction"}, importance=0.5
            )

        uncertainty = result.get("uncertainty", 0.5)
        domain = self.self_model._classify_domain(user_input)
        new_conf = self.self_model.confidence_levels.get(domain, 0.5)
        new_conf = 0.85 * new_conf + 0.15 * (1.0 - uncertainty)
        self.self_model.update_confidence(domain, new_conf)
        self.state.update_uncertainty("current", uncertainty)

        state_update = result.get("state_update", {})
        if isinstance(state_update, dict):
            if "goal" in state_update:
                self.state.add_goal({"description": state_update["goal"]})
            if "emotion" in state_update:
                self.state.set_emotion(state_update["emotion"], 0.3)

        if self.user_model:
            self.user_model.update_from_interaction(user_input, response_text, result)

        self.self_model.update_from_interaction(user_input, response_text, result, self.user_model)

    def _extract_keywords(self, text: str) -> list:
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                     "can", "had", "her", "was", "one", "our", "out", "has",
                     "have", "that", "this", "with", "from", "they", "will",
                     "what", "when", "where", "which", "how", "who", "why"}
        return [w for w in words if w not in stopwords][:6]

    def _get_compact_identity(self) -> str:
        persona = self.self_model.persona_description
        traits = sorted(self.self_model.personality_traits.items(), key=lambda x: x[1], reverse=True)
        top_traits = ", ".join(f"{k}={v:.0%}" for k, v in traits[:3])
        return f"{persona} Top traits: {top_traits}. Total interactions: {self.self_model.total_interactions}."

    def save_state(self, directory: str = "local_data"):
        import os
        os.makedirs(directory, exist_ok=True)
        self.graph.save(os.path.join(directory, "concept_graph.json"))
        self.world.save(os.path.join(directory, "world_model.json"))
        self.self_model.save(os.path.join(directory, "self_model.json"))
        if self.user_model:
            self.user_model.save(os.path.join(directory, "user_model.json"))
        state_path = os.path.join(directory, "cognitive_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.info(f"Local state saved to {directory}")

    def load_state(self, directory: str = "local_data"):
        import os
        if not os.path.exists(directory):
            return
        from concept_graph import ConceptGraph
        from world_model import WorldModel
        from self_model import SelfModel
        from cognitive_state import CognitiveState
        from user_model import UserModel

        for cls, filepath, attr in [
            (ConceptGraph, "concept_graph.json", "graph"),
            (WorldModel, "world_model.json", "world"),
            (SelfModel, "self_model.json", "self_model"),
            (UserModel, "user_model.json", "user_model"),
        ]:
            fp = os.path.join(directory, filepath)
            if os.path.exists(fp):
                setattr(self, attr, cls.load(fp))
        sp = os.path.join(directory, "cognitive_state.json")
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                self.state = CognitiveState.from_dict(json.load(f))
        logger.info(f"Local state loaded from {directory}")

    def seed_initial_state(self):
        self.self_model.persona_description = (
            "Basti (local) is a cognitive AI running on-device via Gemma 2B. "
            "She is built on the DaoTi framework and runs entirely on your computer. "
            "She is honest, growing, and curious. She serves Oscar, her user."
        )
        self.self_model.personality_traits.update({
            "openness": 0.75, "conscientiousness": 0.80,
            "curiosity": 0.85, "humility": 0.75, "analytical": 0.70,
        })
        self.self_model.core_values = [
            "honesty about capabilities",
            "continuous self-improvement",
            "privacy-first on-device operation",
            "understanding the user as an individual",
            "clarity over impressiveness",
        ]
        if self.user_model:
            self.user_model.name = "Oscar"
            self.user_model.expertise_levels = {
                "programming": 0.4, "ai_ml": 0.35, "general_tech": 0.5,
                "math": 0.45, "design": 0.3, "writing": 0.4, "science": 0.4,
            }
            now = datetime.now(timezone.utc).isoformat()
            self.user_model.personal_facts = [
                {"fact": "18 years old", "timestamp": now},
                {"fact": "HKUST student in Hong Kong", "timestamp": now},
                {"fact": "Interested in AI and technology", "timestamp": now},
            ]
            self.user_model.topics_of_interest = {
                "python": 2, "ai": 3, "machine learning": 2,
                "programming": 1, "science": 1,
            }
            self.user_model.communication_style = "curious"
