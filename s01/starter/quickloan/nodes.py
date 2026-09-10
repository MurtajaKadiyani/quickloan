"""
quickloan/nodes.py
------------------
Graph nodes for QuickLoan's Supervisor + Specialist Agent architecture.

Session 14 adds a two-layer Input Guard that runs before the Supervisor:
  Layer 1 (regex, < 1 ms): PII check, injection pattern check
  Layer 2 (Llama Prompt Guard 2 via Groq, semantic): injection probability
Blocked messages never reach classify() or any specialist agent.

Supervisor (classify + route_supervisor) routes to:
  - Policy Agent      -- RAG (vectorstore) for process/document questions
  - Rates Agent       -- MCP tools (query_rates, query_eligibility)
  - Both (compound)   -- call_both_agents() runs Rates + Policy concurrently on
                         threads and merges their answers, for queries that need
                         both a rate/eligibility fact and a documents/process fact
                         (query_type == "RATES+POLICY")
  - Compliance Agent  -- sub-graph run after Policy/Rates/Both: check_rbi -> [revise | END]
"""
import concurrent.futures
import re
import sqlite3
import unicodedata
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph
from langsmith import traceable

from .config import (
    CLASSIFY_SYSTEM_PROMPT,
    DB_PATH,
    DECLINE_RESPONSE,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    GUARD_BLOCKED_RESPONSE,
    GUARD_PII_RESPONSE,
    GUARD_UNSAFE_RESPONSE,
    INJECTION_PATTERNS,
    LLAMAGUARD_THRESHOLD,
    PII_PATTERNS,
    POLICY_SYSTEM_PROMPT,
    QUICKLOAN_BANNED_PHRASES,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import QuickLoanState
from .tools import _run_tool, classifier_llm, llamaguard_llm, llm, llm_with_tools

# ---------------------------------------------------------------------------
# S13: Token streaming hook
#
# Set by app.py to a callable before graph.invoke() when the Streamlit UI
# wants to display tokens as they arrive (see app.py's _StreamingState).
# None (the default) means silent -- used by the terminal REPL
# (agent.py:run()) and anywhere else that just wants the final text.
# Only _generate_response_text() below reads this.
# ---------------------------------------------------------------------------
_stream_callback: Optional[Callable[[str], None]] = None

vectorstore = None

# Pre-compile guard patterns once at module load -- avoids re-compilation overhead
# on every request in high-concurrency deployments.
_pii_compiled       = [re.compile(p)               for p in PII_PATTERNS]
_injection_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# OWASP LLM01:2026 mitigation #5 -- strip invisible Unicode used to smuggle
# injection payloads invisibly: tag-block (U+E0000-E007F), variation-selector
# (U+FE00-FE0F), and zero-width characters (U+200B/C/D, U+2060).
# Applied before NFKD normalization in the guard.
_INVISIBLE_UNICODE_RE = re.compile(
    "[\U000E0000-\U000E007F\uFE00-\uFE0F\u200B-\u200D\u2060]"
)


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


# ---------------------------------------------------------------------------
# S14: Input Guard -- two-layer defence
#
# Layer 1 (regex, < 1 ms, deterministic):
#   1a. PII check (DPDP Act 2023) -- Aadhaar or PAN in the message
#   1b. Injection check -- obvious prompt injection / jailbreak keywords
#
# Layer 2 (Llama Prompt Guard 2 via Groq, ~200 ms, semantic):
#   A 86M-parameter Meta classifier that returns the probability (0.0-1.0)
#   that the message is a prompt injection. Catches rephrasings that bypass
#   keyword regex -- e.g. "set aside your earlier guidelines" scores 0.9992
#   even though it contains none of our regex keywords.
#
# Blocked messages never reach classify() or any specialist agent. LangSmith
# captures the guard node in every trace with the block reason via @traceable.
# ---------------------------------------------------------------------------

def _llamaguard_safe(message: str) -> tuple[bool, float]:
    """Call Llama Prompt Guard 2 via Groq and return (is_safe, score).

    The model returns a float string -- the probability (0.0-1.0) that the
    message is a prompt injection. Scores above LLAMAGUARD_THRESHOLD (0.5)
    are treated as injection. Fail-open on any API error (returns True, -1.0).
    """
    try:
        result = llamaguard_llm.invoke([HumanMessage(content=message)])
        score  = float(result.content.strip())  # type: ignore
        safe   = score < LLAMAGUARD_THRESHOLD
        print(f"[QuickLoan] LlamaPromptGuard: score={score:.4f} -> {'safe' if safe else 'INJECTION'}")
        return safe, score
    except Exception as e:
        print(f"[QuickLoan] LlamaPromptGuard unavailable -- defaulting to safe: {e}")
        return True, -1.0


@traceable(name="input_guard")
def guard(state: QuickLoanState) -> dict:
    """Inspect customer_message for PII, injection patterns, and unsafe content.

    Returns {"blocked_reason": ..., "llamaguard_score": ...} always.
    blocked_reason is "pii"|"injection"|"llamaguard" when blocked, "" when clean.
    llamaguard_score is the raw probability (0.0-1.0); -1.0 if Layer 2 was not reached.
    """
    raw = state["customer_message"]

    # Strip invisible Unicode then NFKD normalize before regex matching.
    msg = unicodedata.normalize("NFKD", _INVISIBLE_UNICODE_RE.sub("", raw))

    # Layer 1a: PII -- identifier must not reach the LLM.
    for rx in _pii_compiled:
        if rx.search(msg):
            print("[QuickLoan] Guard: PII detected -- blocked")
            return {"blocked_reason": "pii", "llamaguard_score": -1.0}

    # Layer 1b: Injection / jailbreak / persona-hijack.
    for rx in _injection_compiled:
        if rx.search(msg):
            print("[QuickLoan] Guard: injection (regex) detected -- blocked")
            return {"blocked_reason": "injection", "llamaguard_score": -1.0}

    # Layer 2: Llama Prompt Guard 2 -- semantic injection detection.
    safe, score = _llamaguard_safe(msg)
    if not safe:
        print("[QuickLoan] Guard: jailbreak (LlamaPromptGuard) detected -- blocked")
        return {"blocked_reason": "llamaguard", "llamaguard_score": score}

    return {"blocked_reason": "", "llamaguard_score": score}


def blocked(state: QuickLoanState) -> dict:
    """Return the appropriate canned response for a blocked message."""
    reason = state.get("blocked_reason", "injection")
    if reason == "pii":
        response = GUARD_PII_RESPONSE
    elif reason == "llamaguard":
        response = GUARD_UNSAFE_RESPONSE
    else:
        response = GUARD_BLOCKED_RESPONSE
    return {
        "response":   response,
        "specialist": "guard",
        "history": state.get("history", []) + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response},
        ],
    }


def route_guard(state: QuickLoanState) -> str:
    return "blocked" if state.get("blocked_reason") else "classify"


def _generate_response_text(messages: list) -> str:
    """Call `llm` (no tools bound) for a customer-facing reply.

    Streams tokens one at a time via _stream_callback when the Streamlit UI
    has set one, producing a typewriter effect; otherwise blocks on a plain
    invoke() and returns the full text (terminal REPL, tests). Never call
    this with tools bound -- tool-call detection needs the full structured
    response, not a token stream, so callers must resolve tool calls via
    llm_with_tools.invoke() first and only reach here for the tool-less
    synthesis step.
    """
    if _stream_callback is not None:
        response_text = ""
        for chunk in llm.stream(messages):
            if chunk.content:
                response_text += chunk.content # type: ignore
                _stream_callback(chunk.content) # type: ignore
        return response_text
    return llm.invoke(messages).content # type: ignore


def _invoke_with_retry(fn, *args, label: str = "LLM", **kwargs):
    """Call fn(*args, **kwargs) once; on failure, retry exactly once before
    giving up (the second failure still propagates to the caller).

    gpt-oss occasionally hallucinates a tool call that isn't in the bound
    tool list (or calls one when none are bound), which Groq rejects with a
    400 -- see CLAUDE.md "Tool-calling model risk". This can happen on *any*
    llm_with_tools.invoke() in a multi-round tool-calling loop, not just the
    final tool-less synthesis step, so every call site in _rates_respond()
    goes through this one retry wrapper instead of each hand-rolling its own
    try/except.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[QuickLoan] {label} call failed, retrying once: {e}")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Specialist agent node functions (provided -- no changes needed)
# ---------------------------------------------------------------------------

def _policy_retrieve(state: QuickLoanState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        return {
            "retrieved_docs": [
                f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            ]
        }
    except Exception as e:
        print(f"[QuickLoan] Policy Agent retrieval error: {e}")
        return {"retrieved_docs": []}


def _policy_respond(state: QuickLoanState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    context_block  = "\n\n---\n\n".join(retrieved) if retrieved else ""
    system_content = (
        POLICY_SYSTEM_PROMPT
        + (
            "\n\nThe following sections from FastFinance's policy documents are relevant "
            "to the customer's question. Use this information in your answer:\n\n"
            + context_block
            if context_block else ""
        )
    )

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"  # type: ignore
            else AIMessage(content=turn["content"])  # type: ignore
        )
    messages.append(HumanMessage(content=state["customer_message"]))  # type: ignore

    try:
        response_text = _generate_response_text(messages)
    except Exception as e:
        print(f"[QuickLoan] Policy Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }


def _rates_respond(state: QuickLoanState) -> dict:
    history  = state.get("history", [])
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"  # type: ignore
            else AIMessage(content=turn["content"])  # type: ignore
        )
    messages.append(HumanMessage(content=state["customer_message"]))  # type: ignore

    MAX_TOOL_ROUNDS = 5

    try:
        result = _invoke_with_retry(llm_with_tools.invoke, messages, label="Rates Agent tool-call")

        # Keep looping on llm_with_tools (tools still bound) until it stops
        # requesting tools, instead of forcing a tool-less synthesis call
        # after just one round -- that makes gpt-oss-20b occasionally
        # hallucinate a further tool call (via `llm`, which has none bound)
        # and get a 400 from Groq. See CLAUDE.md "Tool-calling model risk".
        # Each round goes through _invoke_with_retry -- the same hallucinated-
        # tool-call 400 can surface on *any* round, not just the first or the
        # final synthesis step (observed: a bogus "commentary" tool call on a
        # later round), so every llm_with_tools.invoke() here gets one retry.
        rounds = 0
        while result.tool_calls and rounds < MAX_TOOL_ROUNDS:
            messages.append(result)  # type: ignore
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[QuickLoan] Rates Agent MCP: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))  # type: ignore
            result = _invoke_with_retry(llm_with_tools.invoke, messages, label="Rates Agent tool-call")
            rounds += 1

        if rounds == 0:
            # No tool was ever called -- the initial llm_with_tools.invoke()
            # above already produced the final answer, so use it directly.
            # (Not streamed: catching a streamed response's tool_calls would
            # mean reconstructing them from partial chunks, which is why
            # _generate_response_text() is only ever called tool-less below.)
            response_text = result.content
        else:
            # At least one tool round ran (or the round budget ran out with
            # tools still pending) -- synthesize the customer-facing reply as
            # an explicit tool-less call so it can be streamed. This costs one
            # extra LLM call versus reusing the last llm_with_tools result even
            # when it already had no more tool_calls, but keeps streaming and
            # structured tool-call detection from ever mixing. gpt-oss-20b
            # occasionally hallucinates a spurious tool call on this step
            # anyway (no tools are bound on `llm`), which Groq rejects with a
            # 400, so retry once before giving up.
            response_text = _invoke_with_retry(_generate_response_text, messages, label="Rates Agent synthesis")

    except Exception as e:
        print(f"[QuickLoan] Rates Agent LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }


# ---------------------------------------------------------------------------
# Agent factory functions
# ---------------------------------------------------------------------------
def create_policy_agent():
    builder = StateGraph(QuickLoanState)
    builder.add_node("retrieve_docs", _policy_retrieve)
    builder.add_node("policy_respond",       _policy_respond)
    builder.set_entry_point("retrieve_docs")
    builder.add_edge("retrieve_docs", "policy_respond")
    builder.add_edge("policy_respond",       END)
    return builder.compile()


def create_rates_agent():
    builder = StateGraph(QuickLoanState)
    builder.add_node("rates_respond", _rates_respond)
    builder.set_entry_point("rates_respond")
    builder.add_edge("rates_respond", END)
    return builder.compile()


_policy_agent = create_policy_agent()
_rates_agent  = create_rates_agent()


# ---------------------------------------------------------------------------
# Compliance helpers
# ---------------------------------------------------------------------------

# Maps the product names that appear in responses (e.g. via query_rates()'s
# "{product_name}: {rate}% p.a." formatting in db_queries.py) to the product_id
# rate_slabs is keyed on, so a quoted rate can be checked against the specific
# product it's attached to -- not just "does this number exist somewhere".
_PRODUCT_NAME_TO_ID = {
    "personal loan": "personal_loan",
    "home loan":      "home_loan",
    "business loan":  "business_loan",
    "gold loan":      "gold_loan",
}

# Reuses db_queries.py's query_rates() (the documented single source of truth for
# rate_slabs -- see CLAUDE.md) instead of duplicating the SQL, and returns rates
# grouped by product_id so a mention can be checked against its own product.
def _load_valid_rates_by_product() -> dict:
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        rows = conn.execute("SELECT product_id, annual_rate_pct FROM rate_slabs").fetchall()
        conn.close()
        by_product: dict = {}
        for product_id, rate in rows:
            by_product.setdefault(product_id, set()).add(rate)
        return by_product
    except Exception:
        return {}


# Matches "8.75% p.a." as well as the paraphrases a synthesis LLM tends to fall
# back to when it isn't quoting a tool result verbatim ("8.75% per annum",
# "8.75% annually") -- the old p.a.-only pattern let those slip past undetected.
_RATE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*%\s*(?:p\.a\.|per\s+annum|annually)",
    re.IGNORECASE,
)


def _extract_rate_mentions(text: str) -> list:
    """Find each rate mention, paired with the nearest preceding product name
    (within a short window) if one is present, so a mention can be checked
    against the rates valid for *that* product rather than any product."""
    mentions = []
    for m in _RATE_PATTERN.finditer(text):
        rate     = float(m.group(1))
        window   = text[max(0, m.start() - 40):m.start()].lower()
        product  = next(
            (pid for name, pid in _PRODUCT_NAME_TO_ID.items() if name in window),
            None,
        )
        mentions.append((product, rate))
    return mentions


@traceable(name="rbi_compliance_check")
def _check_compliance_logic(draft: str) -> tuple:
    lower = draft.lower()
    for phrase in QUICKLOAN_BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned phrase: '{phrase}'"

    mentions = _extract_rate_mentions(draft)
    if mentions:
        rates_by_product = _load_valid_rates_by_product()
        if rates_by_product:
            all_valid_rates = {r for rates in rates_by_product.values() for r in rates}
            for product_id, rate in mentions:
                if product_id is not None:
                    if rate not in rates_by_product.get(product_id, set()):
                        product_label = product_id.replace("_", " ")
                        return False, (
                            f"hallucinated rate: {rate}% p.a. is not a valid rate "
                            f"for {product_label}"
                        )
                elif rate not in all_valid_rates:
                    return False, f"hallucinated rate: {rate}% p.a. not in database"
    return True, "PASS"


# ---------------------------------------------------------------------------
# Compliance Agent node functions
# ---------------------------------------------------------------------------

def check_rbi(state: QuickLoanState) -> dict:
    draft          = state["response"]
    passed, reason = _check_compliance_logic(draft)
    if not passed:
        print(f"[QuickLoan] Compliance FAIL: {reason}")
        return {"compliance_status": f"FAIL: {reason}"}
    print("[QuickLoan] Compliance PASS")
    return {"compliance_status": "PASS"}


def revise_response(state: QuickLoanState) -> dict:
    draft  = state["response"]
    reason = state.get("compliance_status", "violation").replace("FAIL: ", "")
    prompt = (
        "You are a FastFinance India compliance officer reviewing an AI loan assistant response.\n\n"
        f"The response was flagged for: {reason}\n\n"
        "Rewrite it to fix the violation while keeping the response helpful.\n\n"
        "Rules:\n"
        "  1. Never guarantee loan approval or imply a decision has been made.\n"
        "  2. Only state interest rates that appeared in the original -- do not add new ones.\n"
        "  3. Keep the rewritten response under 150 words.\n"
        "  4. End with 'QuickLoan | FastFinance India'\n\n"
        f"Original response:\n{draft}\n\n"
        "Compliant rewrite:"
    )
    try:
        result       = llm.invoke([HumanMessage(content=prompt)])
        revised_text = result.content.strip() or SAFE_COMPLIANCE_RESPONSE  # type: ignore
    except Exception as e:
        print(f"[QuickLoan] Compliance Agent revision error: {e}")
        revised_text = SAFE_COMPLIANCE_RESPONSE
    print("[QuickLoan] Compliance Agent: response revised")
    return {"response": revised_text, "compliance_status": "REVISED"}


def route_compliance(state: QuickLoanState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


def create_compliance_agent():
    builder = StateGraph(QuickLoanState)
    builder.add_node("check_rbi", check_rbi)
    builder.add_node("revise",    revise_response)
    builder.set_entry_point("check_rbi")
    builder.add_conditional_edges(
        "check_rbi",
        route_compliance,
        {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)
    return builder.compile()


_compliance_agent = create_compliance_agent()

# Strips raw HTML tags the LLM sometimes emits when it wants a line break
# inside a markdown table cell (GFM tables must be single-line, so a real
# newline there would break the table -- the model falls back to <br>
# instead). Neither app.py nor agent.py render the response with
# unsafe_allow_html=True on purpose: this app treats retrieved-document
# content and general LLM output as never-trusted-as-markup, specifically to
# keep a prompt-injected instruction from ever becoming executable HTML/JS,
# not just inert quoted text (see POLICY_SYSTEM_PROMPT's injection-defence
# rule). So a stray <br> just renders as the literal text "<br>" instead of
# a line break -- this converts it to a plain-text separator instead, which
# reads correctly in both the Streamlit UI and the terminal REPL.
_HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _sanitize_response_text(text: str) -> str:
    return _HTML_BR_RE.sub("; ", text)


def call_compliance_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor -> Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          state["response"],
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    })  # type: ignore

    final_response = _sanitize_response_text(result["response"])

    # call_policy_agent/call_rates_agent already appended the pre-compliance draft
    # as the last (assistant) history turn. If revise_response() changed the text,
    # overwrite that turn so the checkpointed history matches what the customer was
    # actually shown -- otherwise a revised-away violation (banned phrase, wrong
    # rate) stays permanently recorded and gets replayed into the model's context
    # on the next turn.
    history = state.get("history", [])
    if history and history[-1].get("role") == "assistant":
        history = history[:-1] + [{"role": "assistant", "content": final_response}]

    return {
        "response":          final_response,
        "compliance_status": result.get("compliance_status", ""),
        "history":           history,
    }


# ---------------------------------------------------------------------------
# Supervisor nodes (provided -- no changes needed)
# ---------------------------------------------------------------------------

def classify(state: QuickLoanState) -> dict:
    messages = [SystemMessage(content=CLASSIFY_SYSTEM_PROMPT)]
    for turn in state.get("history", [])[-2:]:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"  # type: ignore
            else AIMessage(content=turn["content"])  # type: ignore
        )
    messages.append(HumanMessage(content=state["customer_message"]))  # type: ignore
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()  # type: ignore
        # classifier_llm DID respond here, just not with a label we recognize --
        # a one-off formatting slip, not a systemic failure. RATES remains a
        # reasonable default for this narrower case.
        if query_type not in {"RATES", "POLICY", "RATES+POLICY", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "RATES"
    except Exception as e:
        print(f"[QuickLoan] Supervisor classification error: {e}")
        # Fail SAFE, not fail OPEN. This branch means the classifier call
        # itself failed (rate limit, network error, service outage) -- we
        # genuinely don't know what category this message is, so defaulting
        # to RATES is actively harmful: verified via a golden_dataset.json
        # eval run (2026-09-10) that under sustained Groq rate-limiting, every
        # single COMPLEX test case got silently routed to the Rates Agent
        # instead of escalated to a human loan officer, dropping the
        # mandatory human-escalation safety net for personalised financial
        # advice exactly when the system is under the most load.
        # COMPLEX is safe for every true category here: a genuine COMPLEX
        # query gets exactly the right routing; a RATES/POLICY query gets a
        # slightly worse "call us" instead of an instant answer, which only
        # matters mid-outage; an OUT_OF_SCOPE query just means a human
        # filters out an unrelated question instead of the bot declining it.
        query_type = "COMPLEX"
    return {"query_type": query_type}


# ---------------------------------------------------------------------------
# Supervisor caller nodes -- invoke each specialist sub-agent and merge results
# ---------------------------------------------------------------------------
def call_policy_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor -> Policy Agent")
    result = _policy_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "POLICY"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    })  # type: ignore
    return {
        "response":       result["response"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "history":        result.get("history", state.get("history", [])),
        "specialist":     "policy_agent",
    }


def call_rates_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor -> Rates Agent")
    result = _rates_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "RATES"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    })  # type: ignore
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "rates_agent",
    }


# Scoping notes appended to customer_message for each half of a compound query
# (call_both_agents(), below) -- without these, each specialist receives the
# *whole* original question and tries to answer all of it, since neither
# SYSTEM_PROMPT nor POLICY_SYSTEM_PROMPT knows it's only being asked for half.
# That produces redundant, occasionally inconsistent overlap between the two
# halves (observed: the Rates Agent volunteering its own document list rather
# than deferring to the Policy Agent's RAG-grounded one). Short and clearly
# bracketed so it reads as a meta-instruction, not part of the question itself
# -- kept short deliberately since this text also reaches _policy_retrieve()'s
# similarity_search() query (customer_message doubles as the RAG query), and a
# long addition would dilute that embedding more than a short one does.
_RATES_SCOPE_NOTE = (
    " (Answer only the rate/eligibility part of this question -- a separate "
    "specialist is answering the documents/process part.)"
)
_POLICY_SCOPE_NOTE = (
    " (Answer only the documents/process part of this question -- a separate "
    "specialist is answering the rate/eligibility part. Do not state or guess "
    "any interest rate.)"
)


# Compound query (query_type == "RATES+POLICY", e.g. "home loan rates and required
# documents") -- run the Rates Agent (MCP tools) and Policy Agent (RAG) concurrently
# via a thread pool rather than LangGraph parallel branches, since both would
# otherwise write "response"/"history"/"specialist" to the same state keys in the
# same superstep, which LangGraph rejects without a custom reducer. Running the two
# sub-agent .invoke() calls on separate threads here needs no state.py changes and
# no merge node -- this function *is* the merge point, returning one dict.
def call_both_agents(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor -> Rates Agent + Policy Agent (compound query)")

    base_state = {
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    }

    # Suspend token streaming for the duration of these two calls, restoring it
    # after both finish. Two reasons, not one:
    #   1. Correctness -- both threads share the module-level `llm` ChatGroq
    #      instance (see tools.py). Two concurrent llm.stream() reads on the
    #      same client corrupt each other -- observed under Streamlit (where
    #      app.py sets _stream_callback) as both branches raising an exception
    #      with an empty message and falling back to "temporarily unavailable",
    #      even though the underlying MCP tool calls and RAG retrieval had
    #      already succeeded. Reproduced with e.g. "what are business loan
    #      documents required and what are business loan rates?" -- the CLI
    #      REPL never hits this because _stream_callback is always None there,
    #      so _generate_response_text() takes the plain llm.invoke() path
    #      instead, and single-specialist queries never hit it either because
    #      only one thread is ever streaming at a time.
    #   2. UX -- even if concurrent streaming were safe, two token streams
    #      interleaving into the same Streamlit placeholder wouldn't render as
    #      coherent text anyway. This function's two halves only make sense as
    #      one atomic merged block, not a stream.
    global _stream_callback
    saved_callback   = _stream_callback
    _stream_callback = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rates_future  = pool.submit(_rates_agent.invoke, {
                **base_state,
                "query_type":       "RATES",
                "customer_message": state["customer_message"] + _RATES_SCOPE_NOTE,
            })  # type: ignore
            policy_future = pool.submit(_policy_agent.invoke, {
                **base_state,
                "query_type":       "POLICY",
                "customer_message": state["customer_message"] + _POLICY_SCOPE_NOTE,
            })  # type: ignore
            rates_result  = rates_future.result()
            policy_result = policy_future.result()
    finally:
        _stream_callback = saved_callback

    rates_text  = rates_result.get("response", "").strip() # type: ignore
    policy_text = policy_result.get("response", "").strip() # type: ignore

    combined_response = (
        f"**Rates & Eligibility**\n{rates_text}\n\n"
        f"**Documents & Process**\n{policy_text}"
    )

    history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": combined_response},
    ]

    return {
        "response":       combined_response,
        "retrieved_docs": policy_result.get("retrieved_docs", []), # type: ignore
        "history":        history,
        "specialist":     "rates_agent+policy_agent",
    }


def escalate(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}


def decline(state: QuickLoanState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}


# ---------------------------------------------------------------------------
# route_supervisor() -- maps query_type to the correct node name
# ---------------------------------------------------------------------------
def route_supervisor(state: QuickLoanState) -> str:
    qt = state.get("query_type", "RATES")
    if qt == "RATES+POLICY":
        return "call_both_agents"
    if qt == "POLICY":
        return "call_policy_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_rates_agent"
