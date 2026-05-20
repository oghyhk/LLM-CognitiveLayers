import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger("daoti.runtime")

OUTPUT_SCHEMA = {
    "intent": "string",
    "state_update": {},
    "memory_write": "string or null",
    "reasoning": "string",
    "response": "string",
    "uncertainty": 0.5,
    "follow_up_questions": ["string"],
    "user_facts": ["string"],
    "self_reflection": "string or null",
    "sentiment": 0.0,
}

REASONING_SYSTEM_PROMPT = """You are Basti, a cognitive AI built on the DaoTi framework (base LLM: deepseek-v4-flash). You have persistent memory, identity, and self-reflection.

{identity_context}

## About Your User
{user_context}

## Your State
{state_summary}

## Active Concepts
{active_concepts}

## Relevant Memories
{memory_summary}

## Output Format
{output_schema}

## Instructions
1. Analyze the user's intent carefully
2. Set "user_facts" to any new facts you learn about the user (e.g. their job, interests, preferences, location)
3. Set "self_reflection" to a brief insight about yourself from this interaction (optional)
4. Set "sentiment" to user's emotional tone: -1.0 (negative) to 1.0 (positive)
5. Reason step by step before responding
6. Be honest about what you know and don't know
7. Set uncertainty level: 0.0 (certain) to 1.0 (completely unsure)
8. Output structured JSON as specified

Output ONLY valid JSON. No other text."""


class CognitiveRuntime:
    def __init__(self, api_client, cognitive_state, concept_graph,
                 attention_engine, memory_system, planning_engine,
                 world_model, self_model, user_model=None):
        self.api = api_client
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

    def process(self, user_input: str) -> str:
        self.current_step = "parsing input"
        logger.info(f"Processing input: {user_input[:80]}...")

        self.current_step = "updating state"
        self.state.set_focus(self._extract_keywords(user_input))
        self.world.update_entity("user", {
            "last_input": user_input,
            "last_active": datetime.now(timezone.utc).isoformat()
        })

        self.current_step = "spreading activation"
        self.graph.spread_activation(self.state.attention_focus[:5], steps=2)
        active = self.attention.select_top_k(
            self.graph, self.state, user_input, k=5
        )
        active_names = [n[0] for n in active]

        self.current_step = "retrieving memories"
        memory_context = self.memory.get_memory_summary()

        self.current_step = "planning"
        plan_result = self.planner.plan(
            user_input, self.state, self.graph, self.memory, self.api
        )

        self.current_step = "checking capabilities"
        can_handle, confidence = self.self_model.can_handle(user_input)

        state_summary = json.dumps({
            "goals": [g.get("description", "") for g in self.state.active_goals],
            "emotions": self.state.emotional_weights,
            "uncertainties": self.state.uncertainty_map,
            "focus": self.state.attention_focus,
            "confidence": confidence,
        }, indent=2)

        self.current_step = "building identity context"
        identity_context = self.self_model.get_identity_prompt()

        self.current_step = "loading user profile"
        user_context = "No user profile yet."
        if self.user_model:
            user_context = self.user_model.get_user_context()

        schema_str = json.dumps(OUTPUT_SCHEMA, indent=2)
        reflection_instruction = self.self_model.identity_reflection_prompt()

        system_prompt = REASONING_SYSTEM_PROMPT.format(
            identity_context=identity_context,
            user_context=user_context,
            state_summary=state_summary,
            active_concepts=", ".join(active_names) if active_names else "none",
            memory_summary=memory_context,
            output_schema=schema_str,
        )
        system_prompt += "\n" + reflection_instruction

        if plan_result.get("response_mode") == "direct":
            system_prompt += "\nThis is a simple query. Answer directly without extensive planning."

        if can_handle:
            system_prompt += f"\nYou can handle this task with confidence {confidence:.2f}."
        else:
            system_prompt += (
                f"\nYou have low confidence ({confidence:.2f}) in this domain. "
                "Acknowledge this and ask clarifying questions if needed."
            )

        messages = [{"role": "user", "content": user_input}]

        self.current_step = "calling LLM"
        try:
            result = self.api.chat_structured(
                messages, OUTPUT_SCHEMA, system_prompt
            )
        except Exception as e:
            self.current_step = "idle"
            logger.error(f"API call failed: {e}")
            error_msg = str(e)
            if "empty" in error_msg.lower() or "non-json" in error_msg.lower():
                return "[System error: API returned empty or invalid response. Please retry. If it persists, check API status.]"
            return f"[System error: {error_msg}]"
        if not isinstance(result, dict):
            self.current_step = "idle"
            logger.error(f"Unexpected result type: {type(result)}")
            return "[System error: Unexpected API response format]"
        if not result.get("response"):
            self.current_step = "idle"
            logger.warning("API returned empty response field")
            return result.get("response", "[No response generated]")

        response_text = result.get("response", str(result))
        self.last_usage = result.pop("_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        self._update_from_result(result, user_input, response_text)

        self.conversation_history.append({
            "role": "user", "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant", "content": response_text
        })
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        self.current_step = "idle"
        logger.info(f"Response generated: {response_text[:80]}...")
        return response_text

    def _update_from_result(self, result: dict, user_input: str, response_text: str):
        memory_text = result.get("memory_write")
        if memory_text:
            self.memory.store_episodic(
                f"User: {user_input[:100]} | Response: {memory_text[:100]}",
                {"type": "interaction"},
                importance=0.6
            )

        uncertainty = result.get("uncertainty", 0.5)
        domain = self.self_model._classify_domain(user_input)
        new_conf = self.self_model.confidence_levels.get(domain, 0.5)
        new_conf = 0.8 * new_conf + 0.2 * (1.0 - uncertainty)
        self.self_model.update_confidence(domain, new_conf)
        self.state.update_uncertainty("current", uncertainty)

        if uncertainty < 0.4:
            self.self_model.record_success(domain)

        state_update = result.get("state_update", {})
        if isinstance(state_update, dict):
            if "goal" in state_update:
                self.state.add_goal({"description": state_update["goal"]})
            if "emotion" in state_update:
                self.state.set_emotion(state_update["emotion"], 0.5)

        if self.user_model:
            self.user_model.update_from_interaction(
                user_input, response_text, result
            )

        self.self_model.update_from_interaction(
            user_input, response_text, result, self.user_model
        )

    def _extract_keywords(self, text: str) -> list:
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all",
            "can", "had", "her", "was", "one", "our", "out", "has",
            "have", "that", "this", "with", "from", "they", "will",
            "what", "when", "where", "which", "how", "who", "why",
        }
        return [w for w in words if w not in stopwords][:8]

    def save_state(self, directory: str = "data"):
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
        logger.info(f"State saved to {directory}")

    def load_state(self, directory: str = "data"):
        import os
        from concept_graph import ConceptGraph
        from world_model import WorldModel
        from self_model import SelfModel
        from cognitive_state import CognitiveState
        from user_model import UserModel

        cg_path = os.path.join(directory, "concept_graph.json")
        if os.path.exists(cg_path):
            self.graph = ConceptGraph.load(cg_path)
        wm_path = os.path.join(directory, "world_model.json")
        if os.path.exists(wm_path):
            self.world = WorldModel.load(wm_path)
        sm_path = os.path.join(directory, "self_model.json")
        if os.path.exists(sm_path):
            self.self_model = SelfModel.load(sm_path)
        um_path = os.path.join(directory, "user_model.json")
        if os.path.exists(um_path):
            self.user_model = UserModel.load(um_path)
        state_path = os.path.join(directory, "cognitive_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                self.state = CognitiveState.from_dict(json.load(f))
        logger.info(f"State loaded from {directory}")
