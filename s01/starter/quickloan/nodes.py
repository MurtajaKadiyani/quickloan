"""
quickloan/nodes.py
------------------
Node functions for the QuickLoan graph.

Each node is a plain Python function:
  - Input : the full QuickLoanState (read-only)
  - Output: a dict containing ONLY the keys this node changed
             (LangGraph merges it into the state automatically)
"""
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

from .config import CLASSIFY_SYSTEM_PROMPT, DECLINE_RESPONSE, ESCALATE_RESPONSE, SYSTEM_PROMPT
from .state import QuickLoanState
from .tools import llm, classifier_llm


# ---------------------------------------------------------------------------
# TODO 4 of 5 -- respond node
# ---------------------------------------------------------------------------
# Implement the respond() function so it:
#
#   1. Builds a messages list:
#        messages = [
#            SystemMessage(content=SYSTEM_PROMPT),
#            HumanMessage(content=state["customer_message"]),
#        ]
#
#   2. Calls the LLM inside a try / except block:
#        result = llm.invoke(messages)
#
#   3. On success  → return {"response": result.content}
#      On exception → print the error with a [QuickLoan] prefix
#                      and return a safe fallback string so the
#                      agent never crashes mid-conversation.
#
# ---------------------------------------------------------------------------

def classify(state: QuickLoanState) -> dict:
    """Call the LLM and Classify The response based SIMPLE,COMPLEX,OUT_OF_SCOPE."""
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
        HumanMessage(content=state["customer_message"]),
    ]
 
    try:
       result = classifier_llm.invoke(messages)
       query_type = result.content.strip().upper() # type: ignore
       if query_type not in {"SIMPLE","COMPLEX","OUT_OF_SCOPE"}:
          query_type = "SIMPLE"
    except Exception as e:
        print(f"[QuickLoan] Classification error: {e}")
        query_type = "SIMPLE"
 
    return {"query_type": query_type}

def respond(state: QuickLoanState) -> dict:
    """Call the LLM and return the agent's reply."""
    messages = [
          SystemMessage(content=SYSTEM_PROMPT)
    ]
    history = state.get("history",[])
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"])) # type: ignore
        else:
            messages.append(AIMessage(content=turn["content"])) # type: ignore
    
    messages.append(HumanMessage(content=state["customer_message"])) # type: ignore
    try:
        result = llm.invoke(messages)
        response_text = result.content
        new_history = history + [{"role": "user", "content": state["customer_message"]}, # type: ignore
                             {"role": "assistant", "content": response_text}]
        return {"response": response_text, "history": new_history}
    except Exception as e:
        print(f"[QuickLoan] LLM error: {e}")
        return {"response": "I am temporarily unavailable. Please try again in a moment"}

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
 
def route_query(state: QuickLoanState)->str:
   query_type = state.get("query_type","SIMPLE")
   if query_type == "COMPLEX":
      return "escalate"
   if query_type == "OUT_OF_SCOPE":
      return "decline"
   return "respond"