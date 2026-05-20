# DaoTi Cognitive Architecture — Comparative Overview

This folder contains the full implementation of the DaoTi cognitive architecture with two backends: API-powered and local-powered.

## Four Approaches Compared

| Aspect | Pure LLM | Normal AI Agent | DaoTi API Mode | DaoTi Local Mode |
|--------|----------|----------------|----------------|------------------|
| **State** | Stateless | Session-only | Persistent (saved to disk) | Persistent (saved to disk) |
| **Memory** | None | May have RAG | 4 memory types (episodic, semantic, procedural, reflective) + vector DB | Same 4 memory types + vector DB |
| **Identity** | Prompt only | Basic system prompt | Evolving persona, personality traits, core values, reflection journal | Same identity system, on-device |
| **User Model** | None | None | Learns name, expertise, interests, sentiment, preferences | Same user model |
| **Planning** | Raw prompt | Basic prompts | Structured planning engine with task decomposition, complexity detection | Same planning engine, local-only |
| **Attention** | Full context dump | Full context dump | Top-K concept selection via scoring function (relevance + recency + goal alignment + emotion - decay) | Same attention engine |
| **Concept Graph** | None | None | NetworkX semantic network with spreading activation | Same concept graph |
| **World Model** | None | None | Entity tracking with relationships | Same world model |
| **Self-Awareness** | None | None | Tracks capabilities, limitations, confidence per domain, self-reflection | Same self-model |
| **Runtime Loop** | Prompt → Response | Prompt → Response | 10-step cognitive cycle: input → state update → concept activation → memory retrieval → planning → capability check → identity context → LLM call → state update → respond | Same 10-step cycle, local LLM |
| **Persistence** | Nothing | Nothing | All state, memory, identity, user profile saved on exit | All state saved on exit |
| **Model** | Any LLM | Any LLM + tools | deepseek-v4-flash (via OpenCode Go API) | Gemma 2B (via Ollama, 1.7GB) |
| **Latency (typical)** | 5-15s | 5-30s | 8-30s (network bound) | 2-8s (local inference) |
| **Quality** | High | High | High | Lower (small model) |
| **Privacy** | Depends on provider | Depends on provider | Data sent to API provider | 100% local, no data leaves computer |
| **Cost** | Per-token | Per-token | $10/month OpenCode Go, then per-token | Free (your hardware) |
| **Offline** | No | No | No | Yes |

## How to Run

### API Mode (default)
```
python web_server.py
```
Then open http://127.0.0.1:8080 — uses your OpenCode Go API key.

### Local Mode
Requires Ollama installed. First pull the model:
```
ollama pull gemma:2b
```
Then run:
```
python web_server.py
```
Click the **Local** button in the top header to switch.

### Quick Launch
Double-click `gui.bat` to start the server and open the browser automatically.

## Architecture

```
User Input
    ↓
Intent & State Analyzer
    ↓
Cognitive State Core
    ├── Memory System (SQLite + ChromaDB)
    ├── Concept Graph (NetworkX)
    ├── Planning Engine
    ├── Attention Engine
    ├── World Model
    ├── Self Model
    └── User Model
    ↓
Reasoning Control Loop
    ↓
┌─────────────────┬─────────────────┐
│  API Mode       │  Local Mode     │
│  deepseek-v4    │  Gemma 2B       │
│  (OpenCode Go)  │  (Ollama)       │
└─────────────────┴─────────────────┘
    ↓
Response + State Update + Persistence
```

## Key Insight

The LLM (whether API or local) is only the **reasoning engine**. The architecture — memory, identity, planning, concept graph, self-awareness — is the **cognitive system**. This is what separates it from a plain agent:

> _Intelligence = LLM + Persistent Cognitive System_

## Files

| File | Purpose |
|------|---------|
| `cognitive_runtime.py` | API-mode cognitive loop |
| `local_runtime.py` | Local-mode cognitive loop |
| `api_client.py` | OpenCode Go API client |
| `local_client.py` | Ollama local model client |
| `cognitive_state.py` | Persistent state (goals, emotions, beliefs) |
| `memory_system.py` | SQLite + ChromaDB memory (4 types) |
| `concept_graph.py` | NetworkX semantic network |
| `attention_engine.py` | Top-K concept selection |
| `planning_engine.py` | Task decomposition |
| `world_model.py` | Entity/relationship tracking |
| `self_model.py` | Evolving identity and self-awareness |
| `user_model.py` | User profile that learns over time |
| `web_server.py` | HTTP server with dual-mode support |
| `static/index.html` | Full chat GUI with mode switcher |
| `gui.bat` | One-click launcher |
