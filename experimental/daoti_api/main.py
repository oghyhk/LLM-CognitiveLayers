import os
import sys
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daoti")


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
    }


def seed_user_data(user_model):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    user_model.name = "Oscar"
    user_model.expertise_levels = {
        "programming": 0.4,
        "ai_ml": 0.35,
        "general_tech": 0.5,
        "math": 0.45,
        "design": 0.3,
        "writing": 0.4,
        "science": 0.4,
    }
    user_model.personal_facts = [
        {"fact": "18 years old", "timestamp": now},
        {"fact": "HKUST student in Hong Kong", "timestamp": now},
        {"fact": "Interested in AI and technology", "timestamp": now},
    ]
    user_model.topics_of_interest = {
        "python": 2, "ai": 3, "machine learning": 2,
        "programming": 1, "science": 1,
    }
    user_model.communication_style = "curious"
    logger.info("Seeded initial user data for Oscar")


def build_runtime(config: dict):
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
    state = CognitiveState()
    graph = ConceptGraph()
    attention = AttentionEngine(decay_factor=config["attention_decay"])
    memory = MemorySystem()
    planner = PlanningEngine(
        max_branches=config["planning_branches"],
        max_depth=config["planning_depth"],
    )
    world = WorldModel()
    self_model = SelfModel()
    user_model = UserModel()

    if not os.path.exists(os.path.join("data", "user_model.json")):
        seed_user_data(user_model)

    runtime = CognitiveRuntime(
        api_client=api,
        cognitive_state=state,
        concept_graph=graph,
        attention_engine=attention,
        memory_system=memory,
        planning_engine=planner,
        world_model=world,
        self_model=self_model,
        user_model=user_model,
    )

    if not os.path.exists(os.path.join("data", "user_model.json")):
        try:
            runtime.save_state("data")
            logger.info("Initial seed data saved")
        except Exception as e:
            logger.warning(f"Failed to save seed data: {e}")

    if os.path.exists("data"):
        try:
            runtime.load_state("data")
        except Exception as e:
            logger.warning(f"Could not load saved state: {e}")

    return runtime


def print_banner():
    print(r"""
    ____              _____ _______
   / __ )____ ______ /_  _//_  __/
  / __  / __ `/ ___/  / /   / /
 / /_/ / /_/ (__  )  / /   / /
/_____/\__,_/____/  /_/   /_/

   Basti - Cognitive Architecture
   DaoTi framework | deepseek-v4-flash
   Type /help for commands, /exit to quit
""")


def main():
    config = get_config()

    if not config["api_key"]:
        print("ERROR: API_KEY not set!")
        print("Copy .env.example to .env and add your API key:")
        print("  API_KEY=sk-your-key-here")
        print("  API_BASE_URL=https://api.openai.com/v1  (or your endpoint)")
        sys.exit(1)

    print_banner()
    logger.info(f"Initializing DaoTi with model: {config['api_model']}")
    logger.info(f"API endpoint: {config['api_base_url']}")

    try:
        runtime = build_runtime(config)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        sys.exit(1)

    print("DaoTi ready. You can start chatting.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            runtime.save_state("data")
            runtime.memory.close()
            print("State saved. Goodbye.")
            break
        elif user_input.lower() == "/help":
            print("""
Commands:
  /exit    - Save state and quit
  /help    - Show this help
  /state   - Show current cognitive state
  /mem     - Show recent memories
  /graph   - Show active concepts
  /self    - Show self-model status
  /reset   - Reset cognitive state
  /save    - Save state to disk
""")
            continue
        elif user_input.lower() == "/state":
            state_dict = runtime.state.to_dict()
            print(json.dumps(state_dict, indent=2, default=str))
            continue
        elif user_input.lower() == "/mem":
            print(runtime.memory.get_memory_summary())
            continue
        elif user_input.lower() == "/graph":
            active = runtime.graph.get_active_nodes(threshold=0.2)
            if active:
                for name, data in active[:10]:
                    print(f"  {name}: activation={data.get('activation_score', 0):.2f}")
            else:
                print("No active concepts.")
            continue
        elif user_input.lower() == "/self":
            sm = runtime.self_model
            print(f"Confidence levels: {json.dumps(sm.confidence_levels, indent=2)}")
            print(f"Recent failures: {len(sm.past_failures)}")
            continue
        elif user_input.lower() == "/reset":
            runtime.state.reset()
            print("Cognitive state reset.")
            continue
        elif user_input.lower() == "/save":
            runtime.save_state("data")
            print("State saved to data/")
            continue

        try:
            response = runtime.process(user_input)
            print(f"\nDaoTi: {response}")
        except Exception as e:
            logger.error(f"Error processing: {e}")
            print(f"\nDaoTi: [Error: {e}]")


if __name__ == "__main__":
    main()
