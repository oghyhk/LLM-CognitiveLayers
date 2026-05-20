import networkx as nx
from datetime import datetime, timezone
from typing import Optional
import json
import os


class ConceptGraph:
    RELATION_TYPES = ["related_to", "causes", "supports", "contradicts", "part_of"]

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_concept(self, name: str, embedding: Optional[list] = None):
        if name not in self.graph:
            self.graph.add_node(
                name,
                embedding=embedding,
                activation_score=0.5,
                salience=0.5,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def add_relation(self, source: str, target: str, relation_type: str):
        if relation_type not in self.RELATION_TYPES:
            raise ValueError(f"Invalid relation type: {relation_type}")
        self.add_concept(source)
        self.add_concept(target)
        self.graph.add_edge(source, target, relation_type=relation_type)

    def activate(self, node_name: str, score: float):
        if node_name in self.graph:
            current = self.graph.nodes[node_name].get("activation_score", 0.5)
            self.graph.nodes[node_name]["activation_score"] = min(1.0, max(0.0, score))
            self.graph.nodes[node_name]["timestamp"] = datetime.now(timezone.utc).isoformat()

    def get_active_nodes(self, threshold: float = 0.3) -> list:
        active = []
        for node, data in self.graph.nodes(data=True):
            if data.get("activation_score", 0) >= threshold:
                active.append((node, data))
        return sorted(active, key=lambda x: x[1].get("activation_score", 0), reverse=True)

    def spread_activation(self, seed_nodes: list, decay: float = 0.15, steps: int = 3):
        for node_name in seed_nodes:
            self.activate(node_name, 1.0)
        for step in range(steps):
            factor = 1.0 - decay * (step + 1)
            if factor <= 0:
                break
            nodes = list(self.graph.nodes(data=True))
            updates = {}
            for node_name, _ in nodes:
                predecessors = list(self.graph.predecessors(node_name))
                if not predecessors:
                    continue
                incoming = 0.0
                for pred in predecessors:
                    incoming += self.graph.nodes[pred].get("activation_score", 0.5)
                avg_incoming = incoming / len(predecessors)
                current = self.graph.nodes[node_name].get("activation_score", 0.5)
                updates[node_name] = current + factor * (avg_incoming - current)
            for node_name, new_score in updates.items():
                self.graph.nodes[node_name]["activation_score"] = min(1.0, max(0.0, new_score))

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"name": n, **{k: v for k, v in d.items() if k != "embedding"}}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "relation_type": d.get("relation_type")}
                for u, v, d in self.graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptGraph":
        cg = cls()
        for node in d.get("nodes", []):
            name = node.pop("name")
            cg.graph.add_node(name, **node)
        for edge in d.get("edges", []):
            cg.graph.add_edge(
                edge["source"], edge["target"],
                relation_type=edge.get("relation_type", "related_to")
            )
        return cg

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConceptGraph":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
