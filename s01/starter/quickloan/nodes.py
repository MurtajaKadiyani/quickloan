from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings

from .config import (
    CLASSIFY_SYSTEM_PROMPT, DECLINE_RESPONSE, ESCALATE_RESPONSE,
    EMBED_MODEL, RETRIEVAL_K, SYSTEM_PROMPT, VECTORSTORE_DIR,
)
from .state import QuickLoanState
from .tools import classifier_llm, llm, llm_with_tools, _run_tool

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
            + "\n\nThe following sections from FastFinance India's policy documents are relevant "
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

    try:
        result = llm_with_tools.invoke(messages)

        if result.tool_calls:
            messages.append(result) # type: ignore
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(f"[QuickLoan] Tool: {tc['name']}({tc['args']}) -> {str(tool_output)[:80]}")
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"])) # type: ignore
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