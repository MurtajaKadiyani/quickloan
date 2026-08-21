"""
quickloan/nodes.py
------------------
STARTER FILE -- your task is to implement the three TODO sections below.

Goal
  Refactor the QuickLoan agent into a Supervisor + Specialist Agent architecture.

What is already provided
  - _init_vectorstore(), _policy_retrieve(), _policy_respond(), _rates_respond()
    (the internal node functions for each specialist agent)
  - classify(), escalate(), decline() supervisor utility nodes
  - All imports and state type

Your task
  TODO 1: Implement the agent factory functions create_policy_agent() and
          create_rates_agent(), then instantiate them.
  TODO 2: Implement the supervisor caller nodes call_policy_agent() and
          call_rates_agent() that invoke the sub-agents and merge results.
  TODO 3: Implement route_supervisor() to map query_type to the correct node name.

Run when done
  python -m quickloan.agent   (from inside s10/starter/)
"""
import re
import sqlite3

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
    POLICY_SYSTEM_PROMPT,
    QUICKLOAN_BANNED_PHRASES,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import QuickLoanState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools

vectorstore = None


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
        result        = llm.invoke(messages)
        response_text = result.content
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
        result = llm_with_tools.invoke(messages)

        # Keep looping on llm_with_tools (tools still bound) until it stops
        # requesting tools, instead of forcing a tool-less synthesis call
        # after just one round -- that makes gpt-oss-20b occasionally
        # hallucinate a further tool call (via `llm`, which has none bound)
        # and get a 400 from Groq. See CLAUDE.md "Tool-calling model risk".
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
                print(f"[QuickLoan] Rates Agent synthesis call failed, retrying once: {e}")
                result = llm.invoke(messages)

        response_text = result.content

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
# TODO 1 of 3 -- Implement the agent factory functions
# ---------------------------------------------------------------------------
# create_policy_agent(): StateGraph with retrieve_docs → respond → END
# create_rates_agent():  StateGraph with respond → END
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


def call_compliance_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor → Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          state["response"],
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
    })  # type: ignore

    final_response = result["response"]

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
        if query_type not in {"RATES", "POLICY", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "RATES"
    except Exception as e:
        print(f"[QuickLoan] Supervisor classification error: {e}")
        query_type = "RATES"
    return {"query_type": query_type}


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement call_policy_agent() and call_rates_agent()
# ---------------------------------------------------------------------------
# These are the supervisor's nodes that invoke each specialist sub-agent.
# Each must:
#   1. Print a routing message like: print("[QuickLoan] Supervisor → Policy Agent")
#   2. Call the sub-agent's .invoke({...}) with the current state fields
#   3. Return the specialist's response, history, retrieved_docs, and set "specialist" field
# ---------------------------------------------------------------------------
def call_policy_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor → Policy Agent")
    result = _policy_agent.invoke({
        "customer_message": state["customer_message"],
        "history":          state.get("history", []),
        "response":         "",
        "query_type":       state.get("query_type", "POLICY"),
        "retrieved_docs":   [],
        "specialist":       "",
    })  # type: ignore
    return {
        "response":       result["response"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "history":        result.get("history", state.get("history", [])),
        "specialist":     "policy_agent",
    }


def call_rates_agent(state: QuickLoanState) -> dict:
    print("[QuickLoan] Supervisor → Rates Agent")
    result = _rates_agent.invoke({
        "customer_message": state["customer_message"],
        "history":          state.get("history", []),
        "response":         "",
        "query_type":       state.get("query_type", "RATES"),
        "retrieved_docs":   [],
        "specialist":       "",
    })  # type: ignore
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "rates_agent",
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
# TODO 3 of 3 -- Implement route_supervisor()
# ---------------------------------------------------------------------------
# Map query_type to the correct node name:
#   POLICY       → "call_policy_agent"
#   COMPLEX      → "escalate"
#   OUT_OF_SCOPE → "decline"
#   default      → "call_rates_agent"
# ---------------------------------------------------------------------------
def route_supervisor(state: QuickLoanState) -> str:
    qt = state.get("query_type", "RATES")
    if qt == "POLICY":
        return "call_policy_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_rates_agent"
