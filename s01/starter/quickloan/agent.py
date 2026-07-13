"""
quickloan/agent.py
------------------
Graph construction and the terminal loop.

Run the agent from the session folder:
    cd s01/
    python -m quickloan.agent

Session 1 graph:
    START --> respond --> END
"""
import sqlite3

from langgraph.graph import END, StateGraph
from uuid import uuid4

from .config import CHECKPOINT_DB
from langgraph.checkpoint.sqlite import SqliteSaver
from .nodes import respond
from .state import QuickLoanState
from .config import CHECKPOINT_DB

# ---------------------------------------------------------------------------
# TODO 5 of 5 -- build_graph
# ---------------------------------------------------------------------------
# Implement build_graph() so it:
#
#   1. Creates a StateGraph:
#        builder = StateGraph(QuickLoanState)
#
#   2. Registers the respond node:
#        builder.add_node("respond", respond)
#
#   3. Sets the entry point (first node to run):
#        builder.set_entry_point("respond")
#
#   4. Connects respond → END (the graph exits after one response):
#        builder.add_edge("respond", END)
#
#   5. Compiles and returns the graph:
#        return builder.compile()
#
# ---------------------------------------------------------------------------

def build_graph(checkpointer = None):
    # START --> respond --> END
    builder = StateGraph(QuickLoanState)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond") #START
    builder.add_edge("respond", END) # END
    return builder.compile()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

# Module-level graph instance required by langgraph.json for LangGraph Studio.
# run() uses this directly rather than building a second copy.
graph = build_graph()


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))  # terminal app opts into disk persistence explicit
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}
    print("=" * 55)
    print("  QuickLoan | FastFinance India")
    print("  Type 'quit' to exit")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nQuickLoan: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nQuickLoan: Thank you for choosing FastFinance India. Goodbye!")
            break
        
        # "response": "" is a placeholder to satisfy the TypedDict contract.
        # respond() overwrites it; graph.invoke() returns the full merged state.
        result = _graph.invoke({"customer_message": user_input, "response": ""},config=config) # type: ignore
        print(f"\nQuickLoan: {result['response']}")


if __name__ == "__main__":
    run()
