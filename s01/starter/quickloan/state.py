"""
quickloan/state.py
------------------
The shared state that flows through the LangGraph graph.

Every node reads from this state and writes back a partial update.
Only define the shape here -- no logic.
"""
from typing import TypedDict


class QuickLoanState(TypedDict):
   customer_message : str   # the question the customer typed
   response         : str   # the answer QuickLoan will return
   history: list[dict]
   query_type: str
   retrieved_docs:   list[str]
   compliance_status: str
   specialist:       str
   blocked_reason:    str   # "" = clean; "pii", "injection", or "llamaguard" = blocked by guard
   llamaguard_score: float  # Layer 2 injection probability (0.0-1.0); -1.0 if not reached
