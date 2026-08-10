"""
quickloan/tools.py
------------------
LLM clients and database tool functions for QuickLoan.

Session 5 (PRD US-04): adds query_rate(), query_eligibility(), and calculate_emi()
so the agent looks up live data instead of relying on hardcoded rates in the prompt.
"""
import os

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from . import db_queries
from .config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL_NAME, CLASSIFIER_TEMPERATURE,
    MAX_TOKENS, MODEL_NAME, TEMPERATURE,
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
    max_retries=6,  # openai/gpt-oss-20b sits on an 8000 TPM free-tier cap -- back off and retry through 429s
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=CLASSIFIER_MODEL_NAME,
    temperature=CLASSIFIER_TEMPERATURE,
    max_tokens=CLASSIFIER_MAX_TOKENS,
    max_retries=6,
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
    return db_queries.query_rates(product_id)


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
    return db_queries.query_eligibility(product_id)


@tool
def query_branch(city: str = "all") -> str:
    """Fetch FastFinance India branch contact details from the database.

    Args:
        city: Filter branches by city name, e.g. "Pune", "Mumbai", "Bengaluru".
              Use "all" for every branch (default).

    Returns branch address, phone, and email as a plain-text string.
    """
    return db_queries.query_branch(city)

TOOLS = [
    query_rates,
    query_eligibility,
    query_branch,
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