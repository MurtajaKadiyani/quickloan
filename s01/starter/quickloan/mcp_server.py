"""
quickloan/mcp_server.py
------------------------
Standalone MCP server exposing QuickLoan's FastFinance India database tools
over the MCP protocol (Session 7, PRD US-06 Part 1).

The actual SQL and formatting live in db_queries.py, shared with tools.py's
LangChain @tool functions (query_rates, query_eligibility, query_branch) --
this module just wraps that same logic with @mcp.tool() so the MCP-based
and in-graph tool calls can never drift apart.

Run (must run as a package module -- db_queries.py resolves the DB path via
config.py's package-relative import):
    python -m quickloan.mcp_server

Inspect with MCP Inspector:
    npx @modelcontextprotocol/inspector python -m quickloan.mcp_server
    Open http://localhost:5173 -- query_rates, query_eligibility, and
    query_branch should all appear.
"""

from mcp.server.fastmcp import FastMCP

from . import db_queries

# ---------------------------------------------------------------------------
# Server instantiation
# ---------------------------------------------------------------------------

mcp = FastMCP("quickloan-tools")


# ---------------------------------------------------------------------------
# query_rates -- delegates to db_queries.py (shared with tools.py)
# ---------------------------------------------------------------------------

@mcp.tool()
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


# ---------------------------------------------------------------------------
# query_eligibility -- delegates to db_queries.py (shared with tools.py)
# ---------------------------------------------------------------------------

@mcp.tool()
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


# ---------------------------------------------------------------------------
# query_loan_product -- delegates to db_queries.py (shared with tools.py)
# ---------------------------------------------------------------------------

@mcp.tool()
def query_loan_product(product_id: str = "all") -> str:
    """Fetch FastFinance India loan product terms from the database.

    Covers min/max tenure, max loan amount, and processing fee -- terms that
    query_rates and query_eligibility don't return.

    Args:
        product_id: Which loan's terms to return. Options:
            "personal_loan" -- personal loan terms
            "home_loan"     -- home loan terms
            "business_loan" -- business loan terms
            "gold_loan"     -- gold loan terms
            "all"           -- all products (default)

    Returns formatted product terms as a plain-text string.
    """
    return db_queries.query_loan_product(product_id)


# ---------------------------------------------------------------------------
# query_branch -- delegates to db_queries.py (shared with tools.py)
# ---------------------------------------------------------------------------

@mcp.tool()
def query_branch(city: str = "all") -> str:
    """Fetch FastFinance India branch contact details from the database.

    Args:
        city: Filter branches by city name, e.g. "Pune", "Mumbai", "Bengaluru".
              Use "all" for every branch (default).

    Returns branch address, phone, and email as a plain-text string.
    """
    return db_queries.query_branch(city)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
