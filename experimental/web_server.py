import json
import os
import sys
import logging
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daoti.web")

api_runtime = None
local_runtime = None
current_mode = "api"


def get_config():
    return {
        "api_base_url": os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("API_KEY", ""),
        "api_model": os.getenv("API_MODEL", "deepseek-v4-flash"),
        "api_max_tokens": int(os.getenv("API_MAX_TOKENS", "1024")),
        "api_temperature": float(os.getenv("API_TEMPERATURE", "0.7")),
        "api_timeout": int(os.getenv("API_TIMEOUT", "60")),
        "top_k_concepts": int(os.getenv("TOP_K_CONCEPTS", "5")),
        "attention_decay": float(os.getenv("ATTENTION_DECAY", "0.01")),
        "planning_depth": int(os.getenv("PLANNING_DEPTH", "3")),
        "planning_branches": int(os.getenv("PLANNING_BRANCHES", "3")),
        "memory_retrieval_count": int(os.getenv("MEMORY_RETRIEVAL_COUNT", "10")),
        "host": os.getenv("HOST", "127.0.0.1"),
        "port": int(os.getenv("PORT", "8080")),
        "local_model": os.getenv("LOCAL_MODEL", "gemma:2b"),
        "local_base_url": os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1"),
    }


def build_api_runtime(config):
    from api_client import APIClient
    from cognitive_state import CognitiveState
    from concept_graph import ConceptGraph
    from attention_engine import AttentionEngine
    from memory_system import MemorySystem
    from planning_engine import PlanningEngine
    from world_model import WorldModel
    from self_model import SelfModel
    from user_model import UserModel
    from cognitive_runtime import CognitiveRuntime

    api = APIClient(
        base_url=config["api_base_url"],
        api_key=config["api_key"],
        model=config["api_model"],
        max_tokens=config["api_max_tokens"],
        temperature=config["api_temperature"],
        timeout=config["api_timeout"],
    )
    rt = CognitiveRuntime(
        api_client=api,
        cognitive_state=CognitiveState(),
        concept_graph=ConceptGraph(),
        attention_engine=AttentionEngine(decay_factor=config["attention_decay"]),
        memory_system=MemorySystem(db_path="data/api_memory.db", vector_path="data/api_vectors"),
        planning_engine=PlanningEngine(
            max_branches=config["planning_branches"],
            max_depth=config["planning_depth"],
        ),
        world_model=WorldModel(),
        self_model=SelfModel(),
        user_model=UserModel(),
    )
    if not os.path.exists(os.path.join("data", "user_model.json")):
        _seed_api_user(rt)
    if os.path.exists("data"):
        try:
            rt.load_state("data")
        except Exception as e:
            logger.warning(f"Could not load API state: {e}")
    return rt


def _seed_api_user(rt):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    rt.user_model.name = "Oscar"
    rt.user_model.expertise_levels = {
        "programming": 0.4, "ai_ml": 0.35, "general_tech": 0.5,
        "math": 0.45, "design": 0.3, "writing": 0.4, "science": 0.4,
    }
    rt.user_model.personal_facts = [
        {"fact": "18 years old", "timestamp": now},
        {"fact": "HKUST student in Hong Kong", "timestamp": now},
        {"fact": "Interested in AI and technology", "timestamp": now},
    ]
    rt.user_model.topics_of_interest = {
        "python": 2, "ai": 3, "machine learning": 2,
        "programming": 1, "science": 1,
    }
    rt.user_model.communication_style = "curious"
    logger.info("Seeded API user data for Oscar")


def build_local_runtime(config):
    from local_client import LocalModelClient
    from local_runtime import LocalCognitiveRuntime
    from cognitive_state import CognitiveState
    from concept_graph import ConceptGraph
    from attention_engine import AttentionEngine
    from memory_system import MemorySystem
    from planning_engine import PlanningEngine
    from world_model import WorldModel
    from self_model import SelfModel
    from user_model import UserModel

    local_llm = LocalModelClient(
        base_url=config["local_base_url"],
        model=config["local_model"],
        max_tokens=config["api_max_tokens"] // 2,
        temperature=config["api_temperature"],
    )
    rt = LocalCognitiveRuntime(
        local_client=local_llm,
        cognitive_state=CognitiveState(),
        concept_graph=ConceptGraph(),
        attention_engine=AttentionEngine(decay_factor=config["attention_decay"]),
        memory_system=MemorySystem(db_path="local_data/local_memory.db", vector_path="local_data/local_vectors"),
        planning_engine=PlanningEngine(
            max_branches=2, max_depth=2,
        ),
        world_model=WorldModel(),
        self_model=SelfModel(),
        user_model=UserModel(),
    )
    if not os.path.exists(os.path.join("local_data", "user_model.json")):
        rt.seed_initial_state()
        logger.info("Seeded local initial state")
        try:
            rt.save_state("local_data")
        except Exception as e:
            logger.warning(f"Failed to save local seed: {e}")
    if os.path.exists("local_data"):
        try:
            rt.load_state("local_data")
        except Exception as e:
            logger.warning(f"Could not load local state: {e}")
    return rt


class APIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="static", **kwargs)

    @property
    def active(self):
        return local_runtime if current_mode == "local" else api_runtime

    def do_POST(self):
        global current_mode
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            message = body.get("message", "").strip()
            if not message:
                self._json({"error": "Empty message"}, 400)
                return
            try:
                response = self.active.process(message)
                self._json({
                    "response": response,
                    "usage": self.active.last_usage,
                    "mode": current_mode,
                })
            except Exception as e:
                logger.error(f"Chat error: {e}")
                self._json({"error": str(e)}, 500)
        elif parsed.path == "/api/mode":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            new_mode = body.get("mode", "api")
            if new_mode in ("api", "local"):
                current_mode = new_mode
                self._json({"mode": current_mode, "model": self._model_name()})
            else:
                self._json({"error": "Invalid mode"}, 400)
        elif parsed.path == "/api/reset":
            self.active.state.reset()
            self._json({"status": "ok"})
        else:
            super().do_POST()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._json(self.active.state.to_dict())
        elif parsed.path == "/api/mem":
            self._json({"summary": self.active.memory.get_memory_summary()})
        elif parsed.path == "/api/graph":
            active_nodes = self.active.graph.get_active_nodes(threshold=0.2)
            nodes = [{"name": n, "score": d.get("activation_score", 0)}
                     for n, d in active_nodes[:10]]
            self._json({"nodes": nodes})
        elif parsed.path == "/api/progress":
            self._json({"step": self.active.current_step or "idle"})
        elif parsed.path == "/api/user":
            self._json(self.active.user_model.to_dict() if self.active.user_model else {})
        elif parsed.path == "/api/identity":
            sm = self.active.self_model
            self._json({
                "persona": sm.persona_description,
                "traits": dict(sm.personality_traits),
                "values": sm.core_values,
                "identity_version": sm.identity_version,
                "total_interactions": sm.total_interactions,
                "reflections_count": len(sm.reflection_journal) if hasattr(sm, 'reflection_journal') else 0,
                "recent_reflections": [r["insight"] for r in sm.reflection_journal[-5:]] if hasattr(sm, 'reflection_journal') and sm.reflection_journal else [],
            })
        elif parsed.path == "/api/mode":
            self._json({"mode": current_mode, "model": self._model_name()})
        elif parsed.path.startswith("/api/"):
            self._json({"error": "Not found"}, 404)
        else:
            super().do_GET()

    def _model_name(self):
        if current_mode == "local":
            return local_runtime.llm.model if local_runtime else "local"
        return api_runtime.api.model if api_runtime else "api"

    def _json(self, data, code=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        logger.info("%s %s" % (self.address_string(), format % args))


def main():
    global api_runtime, local_runtime, current_mode
    config = get_config()

    logger.info("Initializing API runtime...")
    api_runtime = build_api_runtime(config)

    logger.info("Initializing local runtime...")
    try:
        local_runtime = build_local_runtime(config)
        logger.info("Local runtime ready (Gemma 2B via Ollama)")
    except Exception as e:
        logger.error(f"Local runtime failed to initialize: {e}")
        local_runtime = None

    current_mode = "api"
    host = config["host"]
    port = config["port"]
    server = ThreadingHTTPServer((host, port), APIHandler)

    local_status = "available" if local_runtime else "unavailable"
    print(f"\n  Basti Web UI running at:  http://{host}:{port}")
    print(f"  API mode: deepseek-v4-flash (online)")
    print(f"  Local mode: {config['local_model']} ({local_status})")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if api_runtime:
            api_runtime.save_state("data")
            api_runtime.memory.close()
        if local_runtime:
            local_runtime.save_state("local_data")
            local_runtime.memory.close()
        server.server_close()


if __name__ == "__main__":
    main()
