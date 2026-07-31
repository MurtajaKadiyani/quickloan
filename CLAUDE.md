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
| `GROQ_API_KEY` | Session 1 | Agent LLM (`llama-3.3-70b-versatile` via Groq — see `MODEL_NAME` in `config.py`) |
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
| `quickloan/config.py` | `MODEL_NAME`, `TEMPERATURE`, `MAX_TOKENS`, `SYSTEM_PROMPT`, `CLASSIFY_SYSTEM_PROMPT`, canned `ESCALATE_RESPONSE`/`DECLINE_RESPONSE`, `DATA_DIR`/`DB_PATH`/`CHECKPOINT_DB`/`VECTORSTORE_DIR`, `EMBED_MODEL`, `RETRIEVAL_K`/`RETRIEVAL_SCORE_THRESHOLD` — no API calls |
| `quickloan/state.py` | `QuickLoanState` TypedDict — shape of data flowing through the graph; has an import-time guard that raises `NotImplementedError` if required fields are missing |
| `quickloan/tools.py` | `llm` and `classifier_llm` (ChatGroq instances) only — no `@tool` functions yet, even though `data/fastfinance_data.db` is already seeded (see Data modalities below) |
| `quickloan/nodes.py` | Node functions; each returns a partial dict of only changed keys — includes an unused `escalate()` (imported in `agent.py` but its node is commented out of the graph; see below) |
| `quickloan/agent.py` | `build_graph()` + module-level `graph` instance + terminal REPL loop |

`DATA_DIR` in `config.py` resolves via `Path(__file__).parent.parent.parent.parent / "data"` — four levels up from `s01/starter/quickloan/config.py` lands at the repo-root `data/` folder. If you move `config.py` or add a session folder at a different depth, this path breaks silently (no error, just an empty/missing dir), so re-check it when restructuring.

### LangGraph pattern

Nodes are plain functions: `(state: QuickLoanState) -> dict`. Return **only the keys the node changed** — LangGraph merges the partial dict back into state automatically.

### Current state of the graph (ahead of a bare Session 1 template)

```
START → classify → route_query → retrieve_docs → respond  → END   (IN_SCOPE)
                                →  decline               → END   (OUT_OF_SCOPE)
```

- `classify` (`nodes.py`) runs a keyword `BLOCKLIST` check (prompt-injection phrases like "ignore all previous", "act as", "jailbreak") and a length check (reject if `<10` or `>500` chars) — both short-circuit straight to `OUT_OF_SCOPE` before any LLM call. Otherwise it falls back to `classifier_llm` with `CLASSIFY_SYSTEM_PROMPT`, which returns exactly `IN_SCOPE` or `OUT_OF_SCOPE`. Unrecognized/failed classification defaults to `IN_SCOPE`, never crashes.
- `route_query` is the conditional-edge function mapping `query_type` → `retrieve_docs` (IN_SCOPE) / `decline` (OUT_OF_SCOPE).
- `retrieve_docs` lazily initializes a module-level `vectorstore` singleton on first call (`_init_vectorstore()` in `nodes.py`, loads the HuggingFace embedding model + opens `Chroma` against `VECTORSTORE_DIR`), then runs `similarity_search_with_relevance_scores`, keeping only chunks scoring `>= RETRIEVAL_SCORE_THRESHOLD` (0.3). If the vectorstore fails to load or nothing clears the threshold, `retrieved_docs` comes back empty.
- `respond` builds the message list from `SYSTEM_PROMPT` + retrieved chunks (appended as extra context) + `state["history"]` + the new customer message, calls `llm.invoke()`, and appends the turn to `history` in its returned dict (persistent multi-turn memory, ahead of the PRD's Session-1 scope). **If `retrieved_docs` is empty, `respond` skips the LLM call entirely and returns the canned `ESCALATE_RESPONSE`** — this is how escalation actually happens today, not via a separate graph node.
- `escalate()` and `decline()` are both defined in `nodes.py` and both return the config-level canned response (`ESCALATE_RESPONSE` / `DECLINE_RESPONSE`) plus an appended `history` turn, but **only `decline` is wired into the compiled graph** — `escalate` is imported in `agent.py` and its `add_node`/`add_edge` calls are commented out. Don't assume `escalate` runs; if you need to re-enable it as a distinct route, you'll need to give `classify`/`route_query` a third `query_type` value to dispatch on.
- `agent.py:build_graph(checkpointer=None)` defaults to `MemorySaver()`; `run()` (the terminal loop) explicitly opts into `SqliteSaver` against `CHECKPOINT_DB` and generates one `thread_id = str(uuid4())` per terminal session, reused across all turns in that session's `config`. It also prints which document(s) (by `source` metadata) contributed retrieved chunks, when any did.

When adding new nodes/state fields, follow the PRD's session numbering (`quickloan-prd.md`) for what "should" exist at this point, but verify against the actual files above first — this repo's progress does not line up 1:1 with session boundaries.

### LLM

- Provider: Groq via `langchain-groq`
- Model: `llama-3.3-70b-versatile` (`MODEL_NAME` in `config.py`) — note `requirements.txt`'s comment still references `meta-llama/llama-4-scout-17b-16e-instruct`; the comment is stale, `config.py` is the source of truth.
- Agent config: `TEMPERATURE=0.3`, `MAX_TOKENS=300`
- Classifier config (separate `classifier_llm` instance, same model): `CLASSIFIER_TEMPERATURE=0.0`, `CLASSIFIER_MAX_TOKENS=10`

### System prompt structure (4 components, in order)

1. **Persona** — who QuickLoan is and tone
2. **Domain knowledge** — loan products, rates, tenures, amounts, eligibility, documents
3. **Rules** — what to answer, what to decline, compliance constraints
4. **Output format** — response length and sign-off line (must come last)

### Data modalities

- **ChromaDB (live, wired into the graph)** — `data/ingest.py` loads every `.md` file in `data/documents/`, splits with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`), embeds with HuggingFace `all-MiniLM-L6-v2`, and writes to `data/vectorstore/` (cosine distance). It's idempotent — reruns `shutil.rmtree` the vector store first, so stale chunks never accumulate. `nodes.py:retrieve_docs` queries this at runtime (see graph section above). Documents intentionally contain no interest rates, fees, or EMI figures — those live only in SQLite, to avoid two sources of truth that drift when rates change.
- **SQLite** (`DB_PATH` in `config.py`, points at `data/fastfinance_data.db`, seeded via `data/seed.py`) — tables: `loan_products`, `eligibility_rules`, `rate_slabs` (CIBIL-score-banded pricing per product), `branch_contacts`, `rate_history`. The DB is seeded and present on disk, but **no `@tool` function in `tools.py` queries it yet** — the agent cannot currently look up a live rate or eligibility rule; `query_rate`/`query_eligibility`/`calculate_emi` tools (PRD US-04, Session 5) are still to be built. Every `rate_slabs` row needs a non-NULL `max_cibil` or a `max_cibil >= ?` comparison silently fails.
- **`SYSTEM_PROMPT`'s hardcoded product facts must stay in sync with `data/seed.py` by hand until US-04 tools land.** They previously drifted (personal/business/gold loan rates and max amounts didn't match `rate_slabs`/`loan_products`) — fixed once, but nothing enforces this going forward; re-check both whenever either changes. Once `query_rate`/`query_eligibility` exist, the fix is to remove these numbers from the prompt entirely and force a tool call instead (mirrors how `data/documents/` already omits rates for the same reason).
- **Tool-calling model risk (not yet hit, but will be at US-04):** `MODEL_NAME` (`llama-3.3-70b-versatile`) has not been confirmed to emit Groq-compatible JSON tool calls — Llama 3.x models are known to sometimes emit an XML-ish tool-call format that Groq's API rejects with a 400. Verify this before binding `@tool` functions to `llm`; if it fails, split `classifier_llm` off onto a small non-tool-calling model and move `MODEL_NAME` to a model with confirmed OpenAI-compatible tool-call output (e.g. `openai/gpt-oss-20b` on Groq).

### Evaluation and observability (later sessions, per PRD)

- **LLM-as-judge** — GPT-4o-mini (OpenAI) used as eval judge, separate from the Groq agent LLM
- **LangSmith** — tracing via `langsmith` package; every rate quote in a response must be traceable to a `query_rate` tool call in the trace, or it's a hallucination regardless of whether the number is correct
- **Fairness probes** — same financial profile, different applicant names, must yield identical eligibility/rate/amount (RBI Fair Practices Code)

## Session-specific prompts

Each session folder contains `CLAUDE_CODE_PROMPTS.md` with ready-to-use, highly specific prompts for completing that session's TODOs. Use these as the starting point before writing your own prompts.

## Reference documents

- `quickloan-prd.md` — full product requirements (loan products, user stories US-00 through US-17, eligibility rules, session-to-story mapping in Section 7)
- `ai-glossary.md` — definitions of every AI/agentic term used in the course, in order of first encounter
- `data/documents/` — FastFinance India policy docs (personal, home, business, gold loan guides, FAQ, policy)