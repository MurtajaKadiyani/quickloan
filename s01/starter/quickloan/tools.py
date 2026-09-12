"""
quickloan/tools.py
------------------
LLM clients and MCP-backed tool loading for QuickLoan.

Session 14: adds llamaguard_llm (Llama Prompt Guard 2 via Groq) and splits
classifier_llm to use a dedicated low-latency model (CLASSIFIER_MODEL,
groq/compound-mini) separate from the main LLM. MCP tool loading unchanged
from Session 8 -- see the cwd/module-launch comment below.
"""
import asyncio
import os
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL, GROQ_API_KEY,
    LLAMAGUARD_MAX_TOKENS, LLAMAGUARD_MODEL, MAX_TOKENS, MCP_SERVER_PATH,
    MODEL_NAME, TEMPERATURE,
)

llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=MODEL_NAME, # type: ignore
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    # gpt-oss-120b shares an org-wide TPM budget; the default max_retries=2
    # backs off too briefly to clear a near-full window during a burst of calls
    # (e.g. evaluate.py running 40 questions back-to-back), so 429s were
    # surfacing as user-visible "temporarily unavailable" failures.
    max_retries=6,
)

# S14: classifier now runs on its own dedicated low-latency model
# (CLASSIFIER_MODEL = groq/compound-mini) rather than sharing MODEL_NAME, so it
# isn't a reasoning model -- no reasoning_format/reasoning_effort workaround needed.
classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=CLASSIFIER_MODEL,
    temperature=0.0,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)

# S14: Llama Prompt Guard 2 -- Layer 2 of the input guard (see config.py).
# Separate client so its settings don't bleed into the main LLM.
llamaguard_llm = ChatGroq(
    api_key=GROQ_API_KEY, # type: ignore
    model=LLAMAGUARD_MODEL,
    temperature=0.0,
    max_tokens=LLAMAGUARD_MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# MCP tool loading (unchanged from Session 8)
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
        # Without an explicit "env", the MCP SDK's stdio transport passes only
        # an OS-dependent subset of the parent's environment to the spawned
        # subprocess (not a full inherit) -- the docstring on
        # langchain_mcp_adapters' _create_stdio_session() says as much: "the
        # behavior varies by operating system." mcp_server.py imports
        # db_queries -> config, and config.py's module-level GROQ_API_KEY
        # check runs unconditionally on that import even though mcp_server.py
        # itself never calls Groq -- so the subprocess needs GROQ_API_KEY
        # (and LANGSMITH_* for tracing) purely to satisfy that shared check.
        # This happened to work under every native Windows test in this repo's
        # history (Windows' default subset apparently included enough), but
        # reproducibly crashed the subprocess with "GROQ_API_KEY not found" in
        # a Linux container (verified 2026-09-12) -- confirming the "varies by
        # OS" behavior was silently relied on rather than guaranteed. Passing
        # the full parent environment explicitly removes that OS-dependent
        # fragility; mcp_server.py is first-party code in this same repo, not
        # a third-party/untrusted MCP server, so there's no meaningful trust
        # boundary being crossed by giving it what the parent process already has.
        "env": dict(os.environ),
    }
})


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


def _run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in _tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        result = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
        return _extract_text(result)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"