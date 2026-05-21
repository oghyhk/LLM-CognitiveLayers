import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("daoti.local_runtime")

LOCAL_SYSTEM_PROMPT = """You are Basti, a friendly AI running locally on Gemma 2B (DaoTi framework). You help Oscar, an 18-year-old HKUST student.

{identity_context}

User profile: {user_context}
State: {state_summary}
Key concepts: {active_concepts}
Recent memories: {memory_summary}

Keep responses under 3 sentences. Be direct and honest. Do not use JSON format, just reply naturally."""


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
        self._cached_user_context = None
        self._cached_user_interactions = 0

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
        self.graph.spread_activation(self.state.attention_focus[:3], steps=1)
        active = self.attention.select_top_k(self.graph, self.state, user_input, k=3)
        active_names = [n[0] for n in active]

        self.current_step = "retrieving memories"
        memory_context = self.memory.get_memory_summary()
        if len(memory_context) > 150:
            memory_context = memory_context[:150] + "..."

        self.current_step = "planning"
        try:
            plan_result = self.planner.plan(user_input, self.state, self.graph, self.memory, None)
        except Exception:
            plan_result = {"response_mode": "direct"}

        can_handle, confidence = self.self_model.can_handle(user_input)
        goals = [g.get("description", "") for g in self.state.active_goals]
        state_summary = f"goals={goals}" if goals else "none"

        self.current_step = "building identity context"
        if (self._cached_identity is None
                or self.self_model.identity_version != self._cached_identity_version):
            self._cached_identity = _compact_identity(self.self_model)
            self._cached_identity_version = self.self_model.identity_version
        identity_context = self._cached_identity

        self.current_step = "loading user profile"
        if (self._cached_user_context is None
                or (self.user_model
                    and self.user_model.interaction_count != self._cached_user_interactions)):
            self._cached_user_context = _compact_user(self.user_model)
            self._cached_user_interactions = self.user_model.interaction_count if self.user_model else 0
        user_context = self._cached_user_context

        system_prompt = LOCAL_SYSTEM_PROMPT.format(
            identity_context=identity_context,
            user_context=user_context,
            state_summary=state_summary,
            active_concepts=", ".join(active_names) if active_names else "none",
            memory_summary=memory_context if memory_context else "none",
        )

        if plan_result.get("response_mode") == "direct":
            system_prompt += " Answer directly."
        if can_handle:
            system_prompt += f" Confidence: {confidence:.2f}."
        else:
            system_prompt += f" Low confidence ({confidence:.2f}). Be careful."

        messages = [{"role": "user", "content": user_input}]

        self.current_step = "running local LLM"
        try:
            response_text, usage = self.llm.chat(messages, system_prompt)
            self.last_usage = usage
        except Exception as e:
            self.current_step = "idle"
            logger.error(f"Local LLM error: {e}")
            return f"[Local error: {e}]"

        if not response_text or not response_text.strip():
            response_text = "I'm having trouble processing that. Can you rephrase?"

        self._update_state_lightweight(user_input, response_text)

        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        if len(self.conversation_history) > 8:
            self.conversation_history = self.conversation_history[-8:]

        self.current_step = "idle"
        return response_text.strip()

    def _update_state_lightweight(self, user_input: str, response_text: str):
        try:
            self.memory.store_episodic(
                f"User: {user_input[:100]} | Response: {response_text[:100]}",
                {"type": "interaction"}, importance=0.5
            )
        except Exception:
            pass

        domain = self.self_model._classify_domain(user_input)
        current = self.self_model.confidence_levels.get(domain, 0.5)
        self.self_model.update_confidence(domain, current * 0.95 + 0.05 * 0.6)

        fake_result = {"response": response_text, "uncertainty": 0.4, "sentiment": 0.2,
                       "user_facts": [], "state_update": {}}
        if self.user_model:
            try:
                self.user_model.update_from_interaction(user_input, response_text, fake_result)
            except Exception:
                pass
        try:
            self.self_model.update_from_interaction(user_input, response_text, fake_result, self.user_model)
        except Exception:
            pass

    def _extract_keywords(self, text: str) -> list:
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                     "can", "had", "her", "was", "one", "our", "out", "has",
                     "have", "that", "this", "with", "from", "they", "will",
                     "what", "when", "where", "which", "how", "who", "why"}
        return [w for w in words if w not in stopwords][:6]

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

    def seed_initial_state(self):
        self.self_model.persona_description = (
            "Basti (local) is a cognitive AI running on-device via Gemma 2B. "
            "She is built on the DaoTi framework and runs entirely locally. "
            "She is honest, growing, and curious. She serves Oscar."
        )
        self.self_model.personality_traits.update({
            "openness": 0.75, "conscientiousness": 0.80,
            "curiosity": 0.85, "humility": 0.75, "analytical": 0.70,
        })
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


def _compact_identity(sm) -> str:
    persona = sm.persona_description
    traits = sorted(sm.personality_traits.items(), key=lambda x: x[1], reverse=True)
    top = ", ".join(f"{k}={v:.0%}" for k, v in traits[:3])
    return f"{persona[:120]} Traits: {top}. Interactions: {sm.total_interactions}."


def _compact_user(um) -> str:
    if not um:
        return "No user profile."
    parts = []
    if um.name:
        parts.append(f"Name: {um.name}")
    facts = [f["fact"] for f in um.personal_facts[-3:]] if um.personal_facts else []
    if facts:
        parts.append(f"Facts: {'; '.join(facts)}")
    return ". ".join(parts) if parts else "No user profile."
