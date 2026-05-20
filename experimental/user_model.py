import json
import os
import re
from datetime import datetime, timezone
from collections import Counter
from typing import Optional


class UserModel:
    def __init__(self):
        self.name: str = ""
        self.first_seen: str = datetime.now(timezone.utc).isoformat()
        self.last_seen: str = self.first_seen
        self.interaction_count: int = 0
        self.preferences: dict[str, str] = {}
        self.expertise_levels: dict[str, float] = {
            "programming": 0.5,
            "ai_ml": 0.5,
            "general_tech": 0.5,
        }
        self.personal_facts: list[dict] = []
        self.topics_of_interest: dict[str, int] = {}
        self.communication_style: str = "neutral"
        self.sentiment_profile: list[float] = []
        self.last_inputs: list[str] = []

    def update_from_interaction(self, user_input: str, assistant_response: str,
                                 response_json: dict):
        self.interaction_count += 1
        self.last_seen = datetime.now(timezone.utc).isoformat()
        self.last_inputs.append(user_input[:200])
        if len(self.last_inputs) > 50:
            self.last_inputs = self.last_inputs[-50:]

        self._extract_name(user_input)
        self._extract_preferences(user_input, assistant_response)
        self._extract_expertise(user_input)
        self._extract_topics(user_input)

        sentiment = response_json.get("sentiment", 0.0)
        if isinstance(sentiment, (int, float)):
            self.sentiment_profile.append(sentiment)
            if len(self.sentiment_profile) > 100:
                self.sentiment_profile = self.sentiment_profile[-100:]

        intent = response_json.get("intent", "")
        user_facts = response_json.get("user_facts", [])
        if isinstance(user_facts, list):
            for fact in user_facts:
                if isinstance(fact, str):
                    self.personal_facts.append({
                        "fact": fact,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    if len(self.personal_facts) > 200:
                        self.personal_facts = self.personal_facts[-200:]

    def _extract_name(self, text: str):
        patterns = [
            r"(?i)(?:my name is|i am|i'm|call me|this is)\s+([A-Z][a-z]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() not in ("not", "just", "sorry", "sure", "also", "here",
                                              "a", "the", "going", "trying", "working"):
                    self.name = candidate
                    break

    def _extract_preferences(self, user_input: str, response: str):
        concise_keywords = ["concise", "short", "brief", "quick", "summarize", "tl;dr"]
        detailed_keywords = ["detail", "thorough", "explain fully", "elaborate", "in depth"]

        if any(w in user_input.lower() for w in concise_keywords):
            self.preferences["response_style"] = "concise"
        elif any(w in user_input.lower() for w in detailed_keywords):
            self.preferences["response_style"] = "detailed"

        if "tone" in user_input.lower():
            tone_match = re.search(r"(?:more|less|be)\s+(\w+)\s*(?:tone|style)?", user_input.lower())
            if tone_match:
                self.preferences["tone"] = tone_match.group(1)

    def _extract_expertise(self, text: str):
        domain_keywords = {
            "programming": ["code", "program", "developer", "coding", "software", "engineer"],
            "ai_ml": ["machine learning", "neural", "model", "train", "dataset", "llm"],
            "math": ["math", "equation", "calculus", "algebra", "theorem"],
            "design": ["design", "ui", "ux", "css", "layout", "color"],
            "writing": ["write", "blog", "article", "essay", "story"],
        }
        for domain, keywords in domain_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                current = self.expertise_levels.get(domain, 0.5)
                self.expertise_levels[domain] = min(1.0, current + 0.02)

    def _extract_topics(self, text: str):
        topic_indicators = [
            "python", "javascript", "react", "ai", "machine learning",
            "database", "api", "frontend", "backend", "devops",
            "security", "testing", "design", "performance", "algorithm",
            "writing", "research", "math", "philosophy", "science",
            "game", "music", "art", "business", "productivity",
        ]
        text_lower = text.lower()
        for topic in topic_indicators:
            if topic in text_lower:
                self.topics_of_interest[topic] = self.topics_of_interest.get(topic, 0) + 1

    def get_user_context(self) -> str:
        parts = []
        if self.name:
            parts.append(f"The user's name is {self.name}.")
        if self.personal_facts:
            recent_facts = [f["fact"] for f in self.personal_facts[-10:]]
            parts.append(f"User personal facts: {'; '.join(recent_facts)}")
        if self.expertise_levels:
            expert = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(
                self.expertise_levels.items(), key=lambda x: x[1], reverse=True
            )[:5])
            parts.append(f"User expertise: {expert}")
        if self.topics_of_interest:
            top_topics = sorted(
                self.topics_of_interest.items(), key=lambda x: x[1], reverse=True
            )[:8]
            parts.append(f"User interests: {', '.join(t for t, _ in top_topics)}")
        if self.sentiment_profile:
            avg_sentiment = sum(self.sentiment_profile) / len(self.sentiment_profile)
            if avg_sentiment > 0.3:
                mood = "generally positive"
            elif avg_sentiment < -0.3:
                mood = "generally frustrated"
            else:
                mood = "neutral"
            parts.append(f"User sentiment: {mood}")
        span = f"{self.interaction_count} interactions since {self.first_seen[:10]}"
        parts.append(f"Relationship: {span}")
        if self.preferences:
            pref_str = "; ".join(f"{k}={v}" for k, v in self.preferences.items())
            parts.append(f"User preferences: {pref_str}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "preferences": self.preferences,
            "expertise_levels": self.expertise_levels,
            "personal_facts": self.personal_facts,
            "topics_of_interest": self.topics_of_interest,
            "communication_style": self.communication_style,
            "sentiment_profile": self.sentiment_profile[-50:] if self.sentiment_profile else [],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserModel":
        um = cls()
        if "name" in d:
            um.name = d["name"]
        if "first_seen" in d:
            um.first_seen = d["first_seen"]
        if "last_seen" in d:
            um.last_seen = d["last_seen"]
        if "interaction_count" in d:
            um.interaction_count = d["interaction_count"]
        if "preferences" in d:
            um.preferences = d["preferences"]
        if "expertise_levels" in d:
            um.expertise_levels = d["expertise_levels"]
        if "personal_facts" in d:
            um.personal_facts = d["personal_facts"]
        if "topics_of_interest" in d:
            um.topics_of_interest = d["topics_of_interest"]
        if "communication_style" in d:
            um.communication_style = d["communication_style"]
        if "sentiment_profile" in d:
            um.sentiment_profile = d["sentiment_profile"]
        return um

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserModel":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
