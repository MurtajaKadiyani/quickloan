"""
quickloan/tools.py
------------------
LLM clients and database tool functions for QuickLoan.

Session 5 (PRD US-04): adds query_rate(), query_eligibility(), and calculate_emi()
so the agent looks up live data instead of relying on hardcoded rates in the prompt.
"""
import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_TEMPERATURE, DB_PATH, MAX_TOKENS,
    MODEL_NAME, TEMPERATURE,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=MODEL_NAME, # type: ignore
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=MODEL_NAME,
    temperature=CLASSIFIER_TEMPERATURE,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)

@tool
def query_rates(product_id: str = "all") -> str:
    """Fetch current FastFinance India interest rates from the database.

    Args:
        product_id: Which loan rates to return. Options:
            "personal_loan" -- personal loan rate slabs by CIBIL score
            "home_loan"     -- home loan rate slabs by CIBIL score
            "business_loan" -- business loan rate slabs by CIBIL score
            "gold_loan"     -- gold loan flat rate
            "all"           -- all products (default)

    Returns formatted rate information as a plain-text string.
    """
    conn  = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    lines = []

    if product_id.lower() == "all":
        rows = conn.execute(
            "SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct "
            "FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id "
            "ORDER BY lp.product_name, rs.min_cibil DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT lp.product_name, rs.min_cibil, rs.max_cibil, rs.annual_rate_pct "
            "FROM rate_slabs rs JOIN loan_products lp ON rs.product_id = lp.product_id "
            "WHERE rs.product_id = ? "
            "ORDER BY rs.min_cibil DESC",
            (product_id,),
        ).fetchall()

    conn.close()

    for name, min_cibil, max_cibil, rate in rows:
        lines.append(f"{name}: {rate:.2f}% p.a. (CIBIL {min_cibil}-{max_cibil})")

    return "\n".join(lines) if lines else f"No rate data found for product: '{product_id}'."


@tool
def query_eligibility(product_id: str = "all") -> str:
    """Fetch FastFinance India loan eligibility criteria from the database.

    Args:
        product_id: Which loan eligibility to return. Options:
            "personal_loan" -- personal loan eligibility rules
            "home_loan"     -- home loan eligibility rules
            "business_loan" -- business loan eligibility rules
            "gold_loan"     -- gold loan eligibility rules
            "all"           -- all products (default)

    Returns formatted eligibility information as a plain-text string.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    if product_id.lower() == "all":
        rows = conn.execute(
            "SELECT lp.product_name, er.min_cibil, er.min_monthly_income, "
            "er.min_age, er.max_age, er.employment_types "
            "FROM eligibility_rules er JOIN loan_products lp ON er.product_id = lp.product_id "
            "ORDER BY lp.product_name"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT lp.product_name, er.min_cibil, er.min_monthly_income, "
            "er.min_age, er.max_age, er.employment_types "
            "FROM eligibility_rules er JOIN loan_products lp ON er.product_id = lp.product_id "
            "WHERE er.product_id = ? "
            "ORDER BY lp.product_name",
            (product_id,),
        ).fetchall()

    conn.close()

    if not rows:
        return f"No eligibility data found for product: '{product_id}'."

    parts = []
    for name, min_cibil, min_income, min_age, max_age, emp_types in rows:
        parts.append(
            f"{name}\n"
            f"  Min CIBIL: {min_cibil} | Min income: Rs. {min_income}/mo | "
            f"Age: {min_age}-{max_age} | {emp_types}"
        )
    return "\n\n".join(parts)

TOOLS = [
    query_rates,
    query_eligibility
]
llm_with_tools = llm.bind_tools(TOOLS)


def _run_tool(tool_name: str, tool_args: dict) -> str:
    _registry = {t.name: t for t in TOOLS}
    if tool_name not in _registry:
        return f"Unknown tool: {tool_name}"
    try:
         return _registry[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"