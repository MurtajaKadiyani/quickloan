# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuickLoan is an AI loan pre-qualification assistant for FastFinance India (NBFC), built as a Launchpad project in the Agentic AI Engineering course (Batch 1, 17 sessions). Each session introduces new capabilities — session folders (`s01/`, `s02/`, …) are released one at a time. The current released session is `s01/`.

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
| `GROQ_API_KEY` | Session 1 | Agent LLM (Llama 4 Scout via Groq) |
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
| `quickloan/config.py` | `MODEL_NAME`, `TEMPERATURE`, `MAX_TOKENS`, `SYSTEM_PROMPT`, `CLASSIFY_SYSTEM_PROMPT`, canned `ESCALATE_RESPONSE`/`DECLINE_RESPONSE`, `DATA_DIR`/`CHECKPOINT_DB` — no API calls |
| `quickloan/state.py` | `QuickLoanState` TypedDict — shape of data flowing through the graph; has an import-time guard that raises `NotImplementedError` if required fields are missing |
| `quickloan/tools.py` | `llm` and `classifier_llm` (ChatGroq instances); grows in later sessions to include `@tool` functions for database queries |
| `quickloan/nodes.py` | Node functions; each returns a partial dict of only changed keys |
| `quickloan/agent.py` | `build_graph()` + module-level `graph` instance + terminal REPL loop |

`DATA_DIR` in `config.py` resolves via `Path(__file__).parent.parent.parent.parent / "data"` — four levels up from `s01/starter/quickloan/config.py` lands at the repo-root `data/` folder. If you move `config.py` or add a session folder at a different depth, this path breaks silently (no error, just an empty/missing dir), so re-check it when restructuring.

### LangGraph pattern

Nodes are plain functions: `(state: QuickLoanState) -> dict`. Return **only the keys the node changed** — LangGraph merges the partial dict back into state automatically.

### Current state of the graph (ahead of a bare Session 1 template)

```
START → classify → route_query →  respond   → END   (SIMPLE)
                                →  escalate  → END   (COMPLEX)
                                →  decline   → END   (OUT_OF_SCOPE)
```

- `classify` (`nodes.py`) runs a keyword `BLOCKLIST` check and a length check (reject if `<10` or `>500` chars) before falling back to `classifier_llm` with `CLASSIFY_SYSTEM_PROMPT`. Unrecognized/failed classification defaults to `SIMPLE`, never crashes.
- `route_query` is the conditional-edge function mapping `query_type` → `respond` / `escalate` / `decline`.
- `respond` builds the message list from `SYSTEM_PROMPT` + `state["history"]` + the new customer message, calls `llm.invoke()`, and appends the turn to `history` in its returned dict (persistent multi-turn memory, ahead of the PRD's Session-1 scope).
- `escalate` / `decline` return the config-level canned responses (`ESCALATE_RESPONSE` / `DECLINE_RESPONSE`) and also append to `history`.
- `agent.py:build_graph(checkpointer=None)` defaults to `MemorySaver()`; `run()` (the terminal loop) explicitly opts into `SqliteSaver` against `CHECKPOINT_DB` and generates one `thread_id = str(uuid4())` per terminal session, reused across all turns in that session's `config`.

When adding new nodes/state fields, follow the PRD's session numbering (`quickloan-prd.md`) for what "should" exist at this point, but verify against the actual files above first — this repo's progress does not line up 1:1 with session boundaries.

### LLM

- Provider: Groq via `langchain-groq`
- Model: `meta-llama/llama-4-scout-17b-16e-instruct`
- Agent config: `TEMPERATURE=0.3`, `MAX_TOKENS=300`
- Classifier config (separate `classifier_llm` instance, same model): `CLASSIFIER_TEMPERATURE=0.0`, `CLASSIFIER_MAX_TOKENS=10`

### System prompt structure (4 components, in order)

1. **Persona** — who QuickLoan is and tone
2. **Domain knowledge** — loan products, rates, tenures, amounts, eligibility, documents
3. **Rules** — what to answer, what to decline, compliance constraints
4. **Output format** — response length and sign-off line (must come last)

### Data modalities (introduced in later sessions, per PRD)

- **SQLite** (`data/fastfinance_data.db`, seeded via `data/seed.py`) — loan product catalog, eligibility rules, CIBIL-score-based rate slabs (every row needs a non-NULL `cibil_max` or the `cibil_max >= ?` comparison silently fails), branch contacts; queried via `@tool` functions
- **ChromaDB** (`data/vectorstore/`, built via `data/ingest.py`) — vector store for RAG over policy documents in `data/documents/`; embeddings via HuggingFace `all-MiniLM-L6-v2`. Documents intentionally contain no interest rates, fees, or EMI figures — those must come from SQLite/the `calculate_emi` tool, never from retrieved text.

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