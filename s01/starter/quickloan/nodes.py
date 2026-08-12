"""
quickloan/nodes.py
------------------
STARTER FILE -- your task is to implement the two TODO sections below.

Goal
  Add an RBI compliance filter to the QuickLoan graph. The filter runs after
  the LLM drafts a response and blocks two classes of violations:
    1. Banned phrases (e.g. "guaranteed approval", "pre-approved")
    2. Hallucinated interest rates not in the FastFinance database

What is already done for you
  - _normalize_for_check()  -- Unicode normalisation so patterns match reliably
  - _BANNED_PATTERN         -- compiled regex over QUICKLOAN_BANNED_PHRASES
  - _load_valid_rates()     -- reads rate_slabs from SQLite
  - _extract_rates()        -- pulls "X% p.a." values out of text
  All other nodes (classify, retrieve_docs, respond, escalate, decline) are
  unchanged from Session 8.

Your task
  TODO 1: implement _check_compliance(draft) -> tuple[bool, str]
  TODO 2: implement check_compliance(state)  -> dict  (the LangGraph node)

Run when done
  python -m quickloan.agent   (from inside s09/starter/)
"""
import re
import sqlite3
import unicodedata

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

from .config import (
    CLASSIFY_SYSTEM_PROMPT,
    DB_PATH,
    DECLINE_RESPONSE,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    QUICKLOAN_BANNED_PHRASES,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import QuickLoanState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools

vectorstore = None

_BANNED_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(p) for p in QUICKLOAN_BANNED_PHRASES),
    re.IGNORECASE,
)


def _normalize_for_check(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in "‐‑‒–—―−":  # hyphen variants + minus sign
        text = text.replace(ch, "-")
    return text.lower()


def _load_valid_rates() -> set:
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        rows = conn.execute("SELECT annual_rate_pct FROM rate_slabs").fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception:
        return set()


def _extract_rates(text: str) -> list:
    matches = re.findall(r"(\d+\.?\d*)\s*%\s*(?:p\.a\.|per\s+annum)", text, re.IGNORECASE)
    return [float(m) for m in matches]


# ---------------------------------------------------------------------------
# TODO 1 of 2 -- Implement _check_compliance()
# ---------------------------------------------------------------------------
# Use _BANNED_PATTERN.search(normalized) to detect banned phrases.
# Then call _extract_rates(normalized) and _load_valid_rates() to catch
# hallucinated interest rates.
#
# Template:
#   @traceable(name="rbi_compliance_check")
#   def _check_compliance(draft: str) -> tuple:
#       normalized = _normalize_for_check(draft)
#       match = _BANNED_PATTERN.search(normalized)
#       if match:
#           return False, f"banned phrase: '{match.group()}'"
#       mentioned_rates = _extract_rates(normalized)
#       if mentioned_rates:
#           valid_rates = _load_valid_rates()
#           if valid_rates:
#               for rate in mentioned_rates:
#                   if rate not in valid_rates:
#                       return False, f"hallucinated rate: {rate}% p.a. not in database"
#       return True, "PASS"
# ---------------------------------------------------------------------------
@traceable(name="rbi_compliance_check")
def _check_compliance(draft: str) -> tuple:
    normalized = _normalize_for_check(draft)
    match = _BANNED_PATTERN.search(normalized)
    if match:
        return False, f"banned phrase: '{match.group()}'"
    mentioned_rates = _extract_rates(normalized)
    if mentioned_rates:
        valid_rates = _load_valid_rates()
        if valid_rates:
            for rate in mentioned_rates:
                if rate not in valid_rates:
                    return False, f"hallucinated rate: {rate}% p.a. not in database"
    return True, "PASS"


# ---------------------------------------------------------------------------
# TODO 2 of 2 -- Implement check_compliance() node
# ---------------------------------------------------------------------------
# Call _check_compliance(state["response"]). If it fails, return
# SAFE_COMPLIANCE_RESPONSE and set compliance_status to "FAIL: <reason>".
# If it passes, return compliance_status "PASS" (leave response unchanged).
# ---------------------------------------------------------------------------
def check_compliance(state: QuickLoanState) -> dict:
    passed, reason = _check_compliance(state["response"])
    if not passed:
        print(f"[QuickLoan] Compliance check FAILED: {reason}")
        return {
            "response": SAFE_COMPLIANCE_RESPONSE,
            "compliance_status": f"FAIL: {reason}",
        }
    return {"compliance_status": "PASS"}


def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[QuickLoan] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def classify(state: QuickLoanState) -> dict:
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper() # type: ignore
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SIMPLE"
    except Exception as e:
        print(f"[QuickLoan] Classification error: {e}")
        query_type = "SIMPLE"
    return {"query_type": query_type}


def retrieve_docs(state: QuickLoanState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs      = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        retrieved = [
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    except Exception as e:
        print(f"[QuickLoan] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: QuickLoanState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nThe following sections from FastFinance's policy documents are relevant "
              "to the customer's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"])) # type: ignore
        else:
            messages.append(AIMessage(content=turn["content"])) # type: ignore
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore

    MAX_TOOL_ROUNDS = 5

    try:
        result = llm_with_tools.invoke(messages)

        # A single question can need several tools (e.g. rates for two
        # products plus a branch lookup). Keep looping on llm_with_tools
        # until it stops requesting tools, instead of forcing a tool-less
        # synthesis call after just one round -- that used to make the model
        # hallucinate further tool calls (via `llm`, which has none bound)
        # and get a 400 from Groq.
        rounds = 0
        while result.tool_calls and rounds < MAX_TOOL_ROUNDS:
            messages.append(result) # type: ignore
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[QuickLoan] MCP tool: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"])) # type: ignore
            result = llm_with_tools.invoke(messages)
            rounds += 1

        if result.tool_calls:
            # Still wants more tools after MAX_TOOL_ROUNDS -- force a final
            # tool-less synthesis from what's been gathered so far. gpt-oss-20b
            # occasionally hallucinates a spurious tool call here anyway (no
            # tools are bound on `llm`), which Groq rejects with a 400, so
            # retry once before giving up.
            try:
                result = llm.invoke(messages)
            except Exception as e:
                print(f"[QuickLoan] Synthesis call failed, retrying once: {e}")
                result = llm.invoke(messages)

        response_text = result.content

    except Exception as e:
        print(f"[QuickLoan] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: QuickLoanState) -> str:
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"