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
# From inside the session folder (e.g. s01/)
cd s01/
python -m quickloan.agent
```

## Testing

```bash
pytest                        # all tests
pytest s01/                   # session-scoped
pytest s01/ -k test_respond   # single test
```

## Environment Variables

| Variable | Required from | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Session 1 | Agent LLM (Llama 4 Scout via Groq) |
| `OPENAI_API_KEY` | Session 6 | GPT-4o-mini used as LLM-as-judge for evaluation |
| `LANGCHAIN_API_KEY` | Session 9 | LangSmith tracing and observability |

## Architecture

### Session folder layout

Each `sNN/` folder is a self-contained Python package:

| File | Role |
|---|---|
| `CLAUDE_CODE_PROMPTS.md` | Ready-to-use Claude Code prompts for completing each session's TODOs |
| `langgraph.json` | LangGraph Studio config — points to `agent.py:graph`, resolves `.env` via `../../.env` (project root) |
| `quickloan/__init__.py` | First file imported; calls `load_dotenv()` so `GROQ_API_KEY` is available before any other module loads |
| `quickloan/config.py` | `MODEL_NAME`, `TEMPERATURE`, `MAX_TOKENS`, `SYSTEM_PROMPT` — no API calls |
| `quickloan/state.py` | `QuickLoanState` TypedDict — shape of data flowing through the graph; has an import-time guard that raises `NotImplementedError` if required fields are missing |
| `quickloan/tools.py` | `llm` (ChatGroq) instance; grows in later sessions to include `@tool` functions for database queries |
| `quickloan/nodes.py` | Node functions (e.g. `respond`); each returns a partial dict of only changed keys |
| `quickloan/agent.py` | `build_graph()` + module-level `graph` instance + terminal REPL loop |

### LangGraph pattern

Nodes are plain functions: `(state: QuickLoanState) -> dict`. Return **only the keys the node changed** — LangGraph merges the partial dict back into state automatically.

Session 1 graph: `START → respond → END`

### LLM

- Provider: Groq via `langchain-groq`
- Model: `meta-llama/llama-4-scout-17b-16e-instruct`
- Config: `TEMPERATURE=0.3`, `MAX_TOKENS=300`

### Session 1 TODOs (in order)

Students complete five TODOs to build the working agent. Each TODO unlocks the next:

| TODO | File | What it does |
|---|---|---|
| 1 | `quickloan/__init__.py` | Import and call `load_dotenv()` so the API key is available at startup |
| 2 | `quickloan/config.py` | Write `SYSTEM_PROMPT` using the 4-component structure (Persona → Domain knowledge → Rules → Output format) |
| 3 | `quickloan/state.py` | Add `customer_message: str` and `response: str` fields to `QuickLoanState` |
| 4 | `quickloan/nodes.py` | Implement `respond()`: build messages list, call `llm.invoke()`, return `{"response": result.content}`, handle exceptions |
| 5 | `quickloan/agent.py` | Implement `build_graph()`: create `StateGraph`, add `respond` node, set entry point, add edge to `END`, compile |

### System prompt structure (4 components, in order)

1. **Persona** — who QuickLoan is and tone
2. **Domain knowledge** — loan products, rates, tenures, amounts, eligibility, documents
3. **Rules** — what to answer, what to decline, compliance constraints
4. **Output format** — response length and sign-off line (must come last)

### Data modalities (introduced in later sessions)

- **SQLite** — loan product catalog, eligibility rules, CIBIL-score-based rate slabs, branch contacts; queried via `@tool` functions
- **ChromaDB** — vector store for RAG over policy documents in `data/documents/`; embeddings via HuggingFace `all-MiniLM-L6-v2`

### Evaluation and observability (later sessions)

- **LLM-as-judge** — GPT-4o-mini (OpenAI) used as eval judge, separate from the Groq agent LLM
- **LangSmith** — tracing via `langsmith` package

## Session-specific prompts

Each session folder contains `CLAUDE_CODE_PROMPTS.md` with ready-to-use, highly specific prompts for completing that session's TODOs. Use these as the starting point before writing your own prompts.

## Reference documents

- `quickloan-prd.md` — full product requirements (loan products, user stories, eligibility rules)
- `ai-glossary.md` — definitions of every AI/agentic term used in the course, in order of first encounter
- `data/documents/` — FastFinance India policy docs (personal, home, business, gold loan guides, FAQ, policy)
