# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuickLoan is an AI loan pre-qualification assistant for FastFinance India (NBFC), built as a Launchpad project in the Agentic AI Engineering course (Batch 1, 17 sessions). Each session introduces new capabilities — session folders (`s01/`, `s02/`, …) are released one at a time by the course, and the current released folder is `s01/`. In practice, this repo's own progress advances by editing `s01/starter/` in place (commit history shows "Session 2 persistent history", "Session 3 LangGraph routing", "Session 4 RAG vectorstore" all landing inside `s01/starter/`, not in new `s02/`/`s03/`/`s04/` folders) — so don't infer the current feature set from which session folders exist on disk. Verify against the actual files, per the [Architecture](#architecture) section below.

**Key constraint:** QuickLoan pre-qualifies only — it never approves or rejects loans. Final approval requires document verification, credit bureau check, and sometimes a field inspection.

## Setup

```bash
pip install -r requirements.txt

# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
# Then fill in your API keys in .env (see Environment Variables below)
```

## Running

```bash
# The package lives at s01/starter/quickloan — run from s01/starter/
cd s01/starter
python -m quickloan.agent
```

## Testing

No test suite exists yet in this repo (no `tests/` folder, no pytest config). The PRD (`quickloan-prd.md`, US-05) specifies pytest with a `conftest.py` providing dummy env vars, an in-memory SQLite fixture, and a mockable LLM-judge fixture (`PYTEST_MOCK_JUDGE=true`) — that lands with the Session 6 evaluation work, not before.

## Environment Variables

| Variable | Required from | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Session 1 | Agent LLM (`openai/gpt-oss-20b` via Groq — see `MODEL_NAME` in `config.py`) plus the classifier LLM (`llama-3.1-8b-instant`, separate model — see `CLASSIFIER_MODEL_NAME`) |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` / `LANGSMITH_TRACING` | Session 4 (basic tracing), full at Session 9 | LangSmith tracing and observability (project: `batch1-quickloan`) |
| `OPENAI_API_KEY` | Session 6 | GPT-4o-mini used as LLM-as-judge for evaluation, never by the agent itself |

## Architecture

### Session folder layout

Each `sNN/` folder is a self-contained Python package. Currently only `s01/starter/` exists:

| File | Role |
|---|---|
| `CLAUDE_CODE_PROMPTS.md` | Ready-to-use Claude Code prompts for completing each session's TODOs |
| `langgraph.json` | LangGraph Studio config — points to `agent.py:graph`, resolves `.env` via `../../.env` (repo root) |
| `quickloan/__init__.py` | First file imported; calls `load_dotenv()` so `GROQ_API_KEY` is available before any other module loads |
| `quickloan/config.py` | `MODEL_NAME`/`TEMPERATURE`/`MAX_TOKENS` (agent) and `CLASSIFIER_MODEL_NAME`/`CLASSIFIER_TEMPERATURE`/`CLASSIFIER_MAX_TOKENS` (classifier, separate model), `SYSTEM_PROMPT`, `CLASSIFY_SYSTEM_PROMPT`, canned `ESCALATE_RESPONSE`/`DECLINE_RESPONSE`, `MCP_SERVER_PATH`, `DATA_DIR`/`DB_PATH`/`CHECKPOINT_DB`/`VECTORSTORE_DIR`, `EMBED_MODEL`, `RETRIEVAL_K`/`RETRIEVAL_SCORE_THRESHOLD` (the latter is defined but currently unused — see graph section below) — no API calls |
| `quickloan/state.py` | `QuickLoanState` TypedDict — shape of data flowing through the graph; has an import-time guard that raises `NotImplementedError` if required fields are missing |
| `quickloan/db_queries.py` | Plain functions (`query_rates`, `query_eligibility`, `query_branch`) — no `@tool`/`@mcp.tool()` decorators, framework-agnostic on purpose. Both `tools.py` (LangChain, in-graph) and `mcp_server.py` (FastMCP, standalone) wrap these instead of duplicating SQL, so the two entry points can't drift apart |
| `quickloan/mcp_server.py` | Standalone MCP server (`FastMCP("quickloan-tools")`) exposing `query_rates`/`query_eligibility`/`query_branch` over MCP via `@mcp.tool()`. Run directly with `python -m quickloan.mcp_server` (STDIO transport), or launched as a subprocess by `tools.py` — see below |
| `quickloan/tools.py` | `llm` and `classifier_llm` (ChatGroq instances), plus a `MultiServerMCPClient` (`langchain_mcp_adapters`) that spawns `mcp_server.py` as a `python -m quickloan.mcp_server` subprocess over STDIO. **At import time** it calls `asyncio.run(_mcp_client.get_tools())` to fetch the MCP tool list, builds `_tool_registry`, and binds the tools to `llm` as `llm_with_tools`. `_run_tool(name, args)` dispatches a call by name via `_tool_registry[...].ainvoke()` and flattens the MCP content-block result to plain text. Because this all runs at import time, simply importing `quickloan.nodes` (which imports `tools.py`) launches the MCP server subprocess — `python -m quickloan.agent` always calls the MCP server, even before the user types anything |
| `quickloan/nodes.py` | Node functions; each returns a partial dict of only changed keys — `escalate()` is now a live, wired-in route (not dead code; see below) |
| `quickloan/agent.py` | `build_graph()` + module-level `graph` instance + terminal REPL loop |

`DATA_DIR` in `config.py` resolves via `Path(__file__).parent.parent.parent.parent / "data"` — four levels up from `s01/starter/quickloan/config.py` lands at the repo-root `data/` folder. If you move `config.py` or add a session folder at a different depth, this path breaks silently (no error, just an empty/missing dir), so re-check it when restructuring.

### LangGraph pattern

Nodes are plain functions: `(state: QuickLoanState) -> dict`. Return **only the keys the node changed** — LangGraph merges the partial dict back into state automatically.

### Current state of the graph (ahead of a bare Session 1 template — now includes Session 7/8 MCP tool-calling)

```
START → classify → route_query → retrieve_docs → respond (may call MCP tools) → END   (SIMPLE)
                                → escalate                                    → END   (COMPLEX)
                                → decline                                     → END   (OUT_OF_SCOPE)
```

- `classify` (`nodes.py`) has **no** keyword/length pre-filter — it calls `classifier_llm` (a separate, smaller model — see LLM section) with `CLASSIFY_SYSTEM_PROMPT` directly and takes the raw response. `CLASSIFY_SYSTEM_PROMPT` now defines **three** categories, not two: `SIMPLE` (factual product/rate/eligibility question), `COMPLEX` (needs personalised assessment, comparison, or EMI calc — routes to a human), `OUT_OF_SCOPE` (unrelated to FastFinance). Unrecognized/failed classification defaults to `SIMPLE`, never crashes.
- `route_query` is the conditional-edge function mapping `query_type` → `retrieve_docs` (`SIMPLE`) / `escalate` (`COMPLEX`) / `decline` (`OUT_OF_SCOPE`).
- `retrieve_docs` lazily initializes a module-level `vectorstore` singleton on first call (`_init_vectorstore()` in `nodes.py`), then runs a plain `similarity_search(k=RETRIEVAL_K)` — **not** `similarity_search_with_relevance_scores`, so `RETRIEVAL_SCORE_THRESHOLD` in `config.py` is currently defined but unused; nothing filters low-relevance chunks anymore. If the vectorstore fails to load, `retrieved_docs` comes back empty.
- `respond` builds the message list from `SYSTEM_PROMPT` (+ retrieved chunks as extra context, if any) + `state["history"]` + the new customer message, then runs an actual **tool-calling loop**: it calls `llm_with_tools.invoke()`; if the result has `tool_calls`, it appends the AI message, runs each call through `_run_tool()` (round-tripping to the MCP server subprocess), appends a `ToolMessage` per result, and re-invokes plain `llm` (retried once — a code comment notes `gpt-oss-20b` occasionally hallucinates a spurious tool call on this synthesis step, which Groq rejects with a 400) to produce the final answer. The turn is appended to `history` in the returned dict (persistent multi-turn memory). `respond` no longer has an empty-`retrieved_docs` → `ESCALATE_RESPONSE` fallback — escalation is now purely `classify`/`route_query` driven.
- `escalate()` and `decline()` are both defined in `nodes.py` and both return the config-level canned response (`ESCALATE_RESPONSE` / `DECLINE_RESPONSE`) plus an appended `history` turn. **Both are wired into the compiled graph now** (`agent.py`'s `add_node`/conditional-edge/`add_edge` calls for `escalate` are live, not commented out) — `query_type == "COMPLEX"` routes there.
- `agent.py:build_graph(checkpointer=None)` defaults to `MemorySaver()`; `run()` (the terminal loop) explicitly opts into `SqliteSaver` against `CHECKPOINT_DB` and generates one `thread_id = str(uuid4())` per terminal session, reused across all turns in that session's `config`. It also prints which document(s) (by `source` metadata) contributed retrieved chunks, when any did, and prints each MCP tool call/result as it happens (`[QuickLoan] Tool: name(args) -> result`).
- **`python -m quickloan.agent` always starts the MCP server**, even before any user input: `agent.py` → `nodes.py` → `tools.py`, and `tools.py` launches `mcp_server.py` as a subprocess and fetches its tool list at import time (see `tools.py` row above).

When adding new nodes/state fields, follow the PRD's session numbering (`quickloan-prd.md`) for what "should" exist at this point, but verify against the actual files above first — this repo's progress does not line up 1:1 with session boundaries.

### LLM

- Provider: Groq via `langchain-groq`
- Agent model: `openai/gpt-oss-20b` (`MODEL_NAME` in `config.py`) — confirmed to emit Groq-compatible JSON tool calls (see "Tool-calling model risk" below, now resolved). `TEMPERATURE=0.3`, `MAX_TOKENS=600` (raised from 300 — tool results folded into a reply need more room). Note `requirements.txt`'s comment on the `langchain-groq` line still references `meta-llama/llama-4-scout-17b-16e-instruct`; the comment is stale, `config.py` is the source of truth.
- Classifier model: **`llama-3.1-8b-instant`** — a genuinely different, smaller model from the agent LLM (`CLASSIFIER_MODEL_NAME` in `config.py`), not just a second `ChatGroq` instance of the same model. `CLASSIFIER_TEMPERATURE=0.0`, `CLASSIFIER_MAX_TOKENS=10`. Kept off `gpt-oss-20b` deliberately — a code comment notes its reasoning-style output breaks `classify()`'s exact-match parse and silently defaults every query to `SIMPLE`.

### System prompt structure (4 components, in order)

1. **Persona** — who QuickLoan is and tone
2. **Domain knowledge** — loan products, rates, tenures, amounts, eligibility, documents
3. **Rules** — what to answer, what to decline, compliance constraints
4. **Output format** — response length and sign-off line (must come last)

### Data modalities

- **ChromaDB (live, wired into the graph)** — `data/ingest.py` loads every `.md` file in `data/documents/`, splits with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`), embeds with HuggingFace `all-MiniLM-L6-v2`, and writes to `data/vectorstore/` (cosine distance). It's idempotent — reruns `shutil.rmtree` the vector store first, so stale chunks never accumulate. `nodes.py:retrieve_docs` queries this at runtime (see graph section above). Documents intentionally contain no interest rates, fees, or EMI figures — those live only in SQLite, to avoid two sources of truth that drift when rates change.
- **SQLite** (`DB_PATH` in `config.py`, points at `data/fastfinance_data.db`, seeded via `data/seed.py`) — tables: `loan_products`, `eligibility_rules`, `rate_slabs` (CIBIL-score-banded pricing per product), `branch_contacts`, `rate_history`. **Now queried live**: `db_queries.py` holds framework-agnostic `query_rates`/`query_eligibility`/`query_branch` functions, wrapped as MCP tools in `mcp_server.py` and bound to the agent LLM in `tools.py` via `MultiServerMCPClient` (see graph section above) — `calculate_emi` is not implemented as a tool. `SYSTEM_PROMPT` now explicitly instructs the model to always call a tool for rates rather than state one from memory. Every `rate_slabs` row still needs a non-NULL `max_cibil` or a `max_cibil >= ?` comparison silently fails.
- **`SYSTEM_PROMPT` no longer hardcodes product rates/amounts** — that drift risk is resolved now that `query_rates`/`query_eligibility` exist as MCP tools; the prompt instead forces a tool call. If you find hardcoded rate numbers reappearing in `SYSTEM_PROMPT`, that's a regression against this design.
- **Tool-calling model risk — resolved.** The original concern (Llama 3.x sometimes emitting an XML-ish tool-call format Groq's API rejects with a 400) was addressed by moving `MODEL_NAME` to `openai/gpt-oss-20b`, which `config.py` marks as "confirmed Groq-compatible tool-call output." The classifier was correspondingly split off onto its own smaller model (`llama-3.1-8b-instant`) rather than sharing the agent model.

### Evaluation and observability (later sessions, per PRD)

- **LLM-as-judge** — GPT-4o-mini (OpenAI) used as eval judge, separate from the Groq agent LLM
- **LangSmith** — tracing via `langsmith` package; every rate quote in a response must be traceable to a `query_rates` tool call in the trace, or it's a hallucination regardless of whether the number is correct
- **Fairness probes** — same financial profile, different applicant names, must yield identical eligibility/rate/amount (RBI Fair Practices Code)

## Session-specific prompts

Each session folder contains `CLAUDE_CODE_PROMPTS.md` with ready-to-use, highly specific prompts for completing that session's TODOs. Use these as the starting point before writing your own prompts.

## Reference documents

- `quickloan-prd.md` — full product requirements (loan products, user stories US-00 through US-17, eligibility rules, session-to-story mapping in Section 7)
- `ai-glossary.md` — definitions of every AI/agentic term used in the course, in order of first encounter
- `data/documents/` — FastFinance India policy docs (personal, home, business, gold loan guides, FAQ, policy)