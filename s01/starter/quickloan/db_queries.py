"""
quickloan/db_queries.py
------------------------
Shared SQLite query logic for FastFinance India's rates, eligibility, and
branch data.

Plain functions, no @tool or @mcp.tool() decorators -- framework-agnostic on
purpose. tools.py (LangChain, wired into the agent graph) and mcp_server.py
(FastMCP, standalone MCP server) both wrap these instead of each keeping
their own copy of the SQL, so the two entry points can't drift apart.
"""
import sqlite3

from .config import DB_PATH


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


def query_branch(city: str = "all") -> str:
    """Fetch FastFinance India branch contact details from the database.

    Args:
        city: Filter branches by city name, e.g. "Pune", "Mumbai", "Bengaluru".
              Use "all" for every branch (default).

    Returns branch address, phone, and email as a plain-text string.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    if city.lower() == "all":
        rows = conn.execute(
            "SELECT city, address, phone, email FROM branch_contacts ORDER BY city"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT city, address, phone, email FROM branch_contacts "
            "WHERE city LIKE ? ORDER BY city",
            (f"%{city}%",),
        ).fetchall()

    conn.close()

    if not rows:
        return f"No FastFinance branches found for city: '{city}'."

    parts = []
    for city_, address, phone, email in rows:
        parts.append(
            f"{city_}\n"
            f"  Address: {address}\n"
            f"  Phone: {phone}  |  Email: {email}"
        )
    return "\n\n".join(parts)
