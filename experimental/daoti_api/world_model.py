import json
import os
from typing import Optional


class WorldModel:
    def __init__(self):
        self.entities = {
            "user": {"name": "user", "state": "active", "preferences": {}},
            "system": {
                "name": "daoti", "state": "running",
                "capabilities": ["reasoning", "planning", "memory"],
                "limitations": ["no internet", "no tool execution"]
            },
            "tasks": [],
            "external_tools": {}
        }
        self.relations = []

    def update_entity(self, entity_name: str, attributes: dict):
        if entity_name in self.entities:
            self.entities[entity_name].update(attributes)
        else:
            self.entities[entity_name] = attributes

    def add_relation(self, source: str, target: str, relation_type: str):
        self.relations.append({
            "source": source,
            "target": target,
            "type": relation_type
        })

    def get_snapshot(self) -> dict:
        return {
            "entities": dict(self.entities),
            "relations": list(self.relations)
        }

    def add_task(self, task: dict):
        self.entities["tasks"].append(task)

    def to_dict(self) -> dict:
        return self.get_snapshot()

    @classmethod
    def from_dict(cls, d: dict) -> "WorldModel":
        wm = cls()
        if "entities" in d:
            wm.entities.update(d["entities"])
        if "relations" in d:
            wm.relations = d["relations"]
        return wm

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
