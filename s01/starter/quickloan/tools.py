"""
quickloan/tools.py
------------------
STARTER FILE -- your task is to implement the three TODO sections below.

Goal
  Connect the QuickLoan agent to the MCP server built in Session 7
  (s07/solution/mcp_server.py) using langchain-mcp-adapters instead of
  hand-written @tool functions. The graph and agent code are unchanged
  from Session 5 -- only how query_rates/query_eligibility are sourced changes.

What is already done for you
  - LLM clients (llm, classifier_llm) are created below
  - MCP_SERVER_PATH points to the Session 7 server
  - _extract_text() and _run_tool()'s docstring/signature are provided

Your task
  TODO 1: Create a MultiServerMCPClient pointing at the Session 7 server
  TODO 2: Load its tools (async, bridged with asyncio.run) and bind them to llm
  TODO 3: Implement _run_tool() to dispatch a call by name and extract the
          plain-text result from the MCP content-block list

Run when done
  python -m quickloan.agent   (from inside s08/starter/)
"""
import asyncio
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL_NAME, CLASSIFIER_TEMPERATURE,
    GROQ_API_KEY, MAX_TOKENS, MCP_SERVER_PATH, MODEL_NAME, TEMPERATURE,
)

llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    # gpt-oss-20b shares an 8000 TPM org-wide budget; the default max_retries=2
    # backs off too briefly to clear a near-full window during a burst of calls
    # (e.g. evaluate.py running 40 questions back-to-back), so 429s were
    # surfacing as user-visible "temporarily unavailable" failures.
    max_retries=6,
)

# Must stay on a separate, smaller model (llama-3.1-8b-instant) from the agent
# LLM -- gpt-oss-20b's reasoning-style output breaks classify()'s exact-match
# parse in nodes.py and silently defaults every query to SIMPLE. See config.py.
classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=CLASSIFIER_MODEL_NAME,
    temperature=CLASSIFIER_TEMPERATURE,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# TODO 1 of 3 -- Create the MultiServerMCPClient
# ---------------------------------------------------------------------------
#   _mcp_client = MultiServerMCPClient({
#       "quickloan": {
#           "transport": "stdio",
#           "command": sys.executable,
#           "args": [str(MCP_SERVER_PATH)],
#       }
#   })
# ---------------------------------------------------------------------------
_mcp_client = MultiServerMCPClient({
    "quickloan": {
        "transport": "stdio",
        "command": sys.executable,
        # mcp_server.py uses "from . import db_queries" -- it only resolves that
        # relative import when launched as a package module (-m quickloan.mcp_server),
        # not as a bare script path, so run it that way with cwd set to the package's
        # parent directory (s01/starter).
        "args": ["-m", "quickloan.mcp_server"],
        "cwd": str(MCP_SERVER_PATH.parent.parent),
    }
})


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Load the server's tools and bind them to the LLM
# ---------------------------------------------------------------------------
#   mcp_tools      = asyncio.run(_mcp_client.get_tools())
#   _tool_registry = {t.name: t for t in mcp_tools}
#   llm_with_tools = llm.bind_tools(mcp_tools)
# ---------------------------------------------------------------------------
mcp_tools      = asyncio.run(_mcp_client.get_tools())
_tool_registry = {t.name: t for t in mcp_tools}
llm_with_tools = llm.bind_tools(mcp_tools)


def _extract_text(result) -> str:
    """MCP tool results come back as a list of content blocks. Provided -- no changes needed."""
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)


# ---------------------------------------------------------------------------
# TODO 3 of 3 -- Implement _run_tool()
# ---------------------------------------------------------------------------
#   def _run_tool(tool_name: str, tool_args: dict) -> str:
#       if tool_name not in _tool_registry:
#           return f"Unknown tool: {tool_name}"
#       try:
#           result = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
#           return _extract_text(result)
#       except Exception as e:
#           return f"Tool error ({tool_name}): {e}"
# ---------------------------------------------------------------------------
def _run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in _tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        result = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
        return _extract_text(result)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"