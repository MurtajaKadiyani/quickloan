"""
app.py
------
STARTER FILE -- Session 13: Streamlit Frontend.

What is already provided (no changes needed):
  - compliance_badge()   returns a display badge string for compliance_status
  - format_route_label() formats the routing info into one line
  - is_escalated()       returns True when specialist == "escalated"
  - _init_session()      initialises graph + session state on first load
  - _sidebar()           renders the sidebar with agent descriptions
  - _render_history()    renders previous chat turns
  - main()               wires everything together

Your task (3 TODOs):
  TODO 1: Implement build_input_state(message)
          Return the dict that graph.invoke() expects as its first argument.
          Fields: customer_message, response, specialist, retrieved_docs, compliance_status

  TODO 2: Implement get_thread_config(thread_id)
          Return {"configurable": {"thread_id": thread_id}}

  TODO 3: In main(), complete the response display block
          Show the response with st.chat_message("assistant")
          Use st.warning() for escalated responses, st.markdown() for others.
          Show the route caption below the message.
          Append response + route_label to session_state.

Run when done:
    streamlit run app.py   (from inside s13/starter/)
"""
import sys
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from quickloan.agent import build_graph  # noqa: E402
import quickloan.nodes as _nodes         # noqa: E402


# ---------------------------------------------------------------------------
# S13: Token streaming — bridges llm.stream() in nodes.py to the Streamlit UI
# ---------------------------------------------------------------------------

# CSS for a genuinely blinking cursor -- the "▌" character on its own is just
# static text that gets replaced on every re-render; wrapping it in this span
# gives it a real CSS animation. Injected once per script run in main().
_CURSOR_CSS = """
<style>
@keyframes qlBlinkCursor { 50% { opacity: 0; } }
.ql-cursor {
    display: inline-block;
    animation: qlBlinkCursor 0.9s step-start infinite;
}
</style>
"""


class _StreamingState:
    """Receives tokens from nodes.py via _nodes._stream_callback and renders
    them incrementally into a Streamlit placeholder, producing a typewriter
    effect with a blinking cursor.

    nodes.py's _generate_response_text() calls this once per token as it
    arrives from llm.stream(). Groq streams in bursts -- a single token can
    already be several words -- so we replay each token one character at a
    time with a small delay between characters, instead of dumping the whole
    token in at once, to get a legible typewriter pace regardless of how
    large the underlying chunks are.
    """

    def __init__(self, placeholder, char_delay: float = 0.0) -> None:
        self._placeholder = placeholder
        self._text  = ""
        self._delay = char_delay  # seconds between characters; 0 = full speed

    def __call__(self, token: str) -> None:
        for ch in token:
            self._text += ch
            # main() overwrites this placeholder with the final clean text
            # (no HTML, no cursor) once graph.invoke() returns.
            self._placeholder.markdown(
                f'{self._text}<span class="ql-cursor">▌</span>',
                unsafe_allow_html=True,
            )
            if self._delay > 0:
                time.sleep(self._delay)


# ---------------------------------------------------------------------------
# TODO 1 of 3 -- Implement build_input_state()
# ---------------------------------------------------------------------------
# Return a dict with these keys (the graph's initial state):
#   "customer_message":  message   (the customer's question)
#   "response":          ""        (empty -- agent fills this in)
#   "specialist":        ""        (empty -- supervisor fills this)
#   "retrieved_docs":    []        (empty list)
#   "compliance_status": ""        (empty -- compliance agent fills this)
#
# Template:
#   def build_input_state(message: str) -> dict:
#       return {
#           "customer_message":  message,
#           "response":          "",
#           "specialist":        "",
#           "retrieved_docs":    [],
#           "compliance_status": "",
#       }
# ---------------------------------------------------------------------------
def build_input_state(message: str) -> dict:
    return {
        "customer_message":  message,
        "response":          "",
        "specialist":        "",
        "retrieved_docs":    [],
        "compliance_status": "",
    }


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement get_thread_config()
# ---------------------------------------------------------------------------
# LangGraph needs a thread ID to keep memory across turns in the same session.
# Return {"configurable": {"thread_id": thread_id}}
#
# Template:
#   def get_thread_config(thread_id: str) -> dict:
#       return {"configurable": {"thread_id": thread_id}}
# ---------------------------------------------------------------------------
def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Already implemented -- no changes needed for these
# ---------------------------------------------------------------------------

def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ RBI Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def format_route_label(result: dict) -> str:
    qt    = result.get("query_type", "—")
    sp    = result.get("specialist", "—")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {qt} → {sp}"
    if badge:
        label += f" | {badge}"
    return label


def is_escalated(result: dict) -> bool:
    return result.get("specialist", "") == "escalated"


def _new_thread() -> None:
    """Start a new, empty conversation with its own thread_id and switch to
    it. The graph/checkpointer (MemorySaver) is shared across every thread
    in this browser session -- each thread_id just gets its own isolated
    slice of memory within it, so switching back to an older thread restores
    that conversation's history exactly as LangGraph checkpointed it."""
    tid = str(uuid4())
    st.session_state.threads[tid] = {"title": "New chat", "messages": [], "routes": []}
    st.session_state.thread_id = tid


def _clear_current_thread() -> None:
    """Wipe the active thread's visible history back to empty, reusing the
    same thread_id and sidebar slot -- unlike _new_thread(), this resets the
    current conversation in place instead of starting a separate one.

    build_input_state() never sets "history", so LangGraph's partial-state
    merge would otherwise leave the checkpointer's old value in place -- the
    chat would look empty but the next turn would still load prior turns as
    context. update_state() explicitly resets it so the clear is real, not
    just cosmetic.
    """
    tid    = st.session_state.thread_id
    thread = st.session_state.threads.get(tid)
    if not thread or not thread["messages"]:
        return  # already empty -- nothing to clear
    st.session_state.threads[tid] = {"title": "New chat", "messages": [], "routes": []}
    try:
        st.session_state.graph.update_state(get_thread_config(tid), {"history": []})
    except Exception as e:
        print(f"[QuickLoan] Could not reset checkpointed history for {tid}: {e}")


def _delete_thread(tid: str) -> None:
    """Remove a conversation from the sidebar entirely. If it was the active
    thread, switch to another existing one (most recently created) or start
    a fresh thread if none remain -- the sidebar should never end up with no
    selectable conversation. The abandoned LangGraph checkpoint for tid is
    left in MemorySaver's memory but is harmless: nothing references that
    thread_id again once its button is gone."""
    st.session_state.threads.pop(tid, None)
    if st.session_state.thread_id == tid:
        remaining = list(st.session_state.threads.keys())
        if remaining:
            st.session_state.thread_id = remaining[-1]
        else:
            _new_thread()


def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph = build_graph(checkpointer=MemorySaver())
    if "threads" not in st.session_state:
        st.session_state.threads = {}
    if "thread_id" not in st.session_state or st.session_state.thread_id not in st.session_state.threads:
        _new_thread()


def _sidebar() -> None:
    with st.sidebar:
        st.header("💰 QuickLoan")
        st.caption("FastFinance AI Loan Assistant")
        st.divider()

        if st.button("➕ New Chat", use_container_width=True):
            # Skip creating another empty thread if the current one was
            # never used -- avoids cluttering the list with blank entries
            # from repeated clicks.
            current = st.session_state.threads.get(st.session_state.thread_id)
            if not current or current["messages"]:
                _new_thread()
                st.rerun()

        if st.button("🧹 Clear Conversation", use_container_width=True):
            _clear_current_thread()
            st.rerun()

        st.divider()
        st.subheader("Conversations")
        # Most recently created thread first.
        for tid in reversed(list(st.session_state.threads.keys())):
            thread = st.session_state.threads[tid]
            active = tid == st.session_state.thread_id
            label  = ("🟢 " if active else "💬 ") + thread["title"]
            col_select, col_delete = st.columns([5, 1])
            with col_select:
                if st.button(
                    label, key=f"thread_btn_{tid}",
                    use_container_width=True, disabled=active,
                ):
                    st.session_state.thread_id = tid
                    st.rerun()
            with col_delete:
                if st.button("🗑️", key=f"thread_del_{tid}", use_container_width=True):
                    _delete_thread(tid)
                    st.rerun()

        st.caption(f"Session: {st.session_state.thread_id[:8]}…")
        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Supervisor** — classifies query\n"
            "- **Rates Agent** — live interest rates via MCP\n"
            "- **Policy Agent** — loan policy via RAG\n"
            "- **Compliance Agent** — RBI rules check\n"
            "- **Escalate** → officer / **Decline** → out-of-scope"
        )

        st.divider()
        st.subheader("Display settings")
        # Stored in session_state so the slider keeps its position across
        # re-runs instead of resetting to the default every time.
        st.session_state["token_delay_ms"] = st.slider(
            "Typing speed (ms per character)",
            min_value=0, max_value=60,
            value=st.session_state.get("token_delay_ms", 15),
            step=5,
            help="Delay between characters while the response streams in. "
                 "0 = full speed (no typewriter effect).",
        )


def _scroll_to_bottom() -> None:
    """Auto-scroll the page to the latest message.

    Streamlit doesn't scroll the page on rerun -- if you've scrolled up to
    reread an earlier turn and then ask a new question, the new answer
    renders below the fold and you'd have to scroll down manually to see it.

    A <script> tag injected via st.markdown(unsafe_allow_html=True) never
    executes -- innerHTML-inserted <script> tags are inert in every browser.
    components.html() renders inside a real iframe document instead, where
    <script> tags do execute; we reach into window.parent because the
    iframe is a separate document from the actual Streamlit page.
    """
    components.html(
        """
        <script>
            var doc = window.parent.document;
            var scroller = doc.querySelector('section.main') || doc.body;
            scroller.scrollTo({top: scroller.scrollHeight, behavior: 'smooth'});
        </script>
        """,
        height=0,
    )


def _render_history() -> None:
    thread   = st.session_state.threads[st.session_state.thread_id]
    messages = thread.get("messages", [])
    routes   = thread.get("routes",   [])
    assistant_idx = 0
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_idx < len(routes):
                st.caption(routes[assistant_idx])
            assistant_idx += 1


def main() -> None:
    st.set_page_config(page_title="QuickLoan | FastFinance", page_icon="💰", layout="wide")
    st.title("💰 QuickLoan | FastFinance")
    st.caption("AI-powered loan assistant — Session 13: Streamlit UI")
    st.markdown(_CURSOR_CSS, unsafe_allow_html=True)

    _init_session()
    _sidebar()
    _render_history()

    prompt = st.chat_input("Ask about loan rates, eligibility, or our policies…")
    if prompt:
        thread = st.session_state.threads[st.session_state.thread_id]
        if not thread["messages"]:
            # First message in this thread -- use it (trimmed) as the
            # sidebar label so each conversation is identifiable at a glance.
            thread["title"] = prompt if len(prompt) <= 40 else prompt[:40].rstrip() + "…"
        thread["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        _scroll_to_bottom()

        # Open the assistant bubble early -- before we have a response -- so
        # the streaming placeholder has somewhere to render into. It gets
        # overwritten token by token by _StreamingState.__call__() as
        # nodes.py's _generate_response_text() streams the reply.
        with st.chat_message("assistant"):
            placeholder = st.empty()

        # Wire up streaming: from this point until we clear it, every token
        # nodes.py emits via llm.stream() calls streamer(token). The
        # try/finally ensures the callback is always cleared, even on
        # exception, so the next turn doesn't reuse a stale streamer.
        char_delay = st.session_state.get("token_delay_ms", 15) / 1000
        _nodes._stream_callback = _StreamingState(placeholder, char_delay=char_delay)
        try:
            result = st.session_state.graph.invoke(
                build_input_state(prompt),
                config=get_thread_config(st.session_state.thread_id),
            )
        finally:
            _nodes._stream_callback = None

        response    = result["response"]
        route_label = format_route_label(result)

        # ---------------------------------------------------------------------------
        # TODO 3 of 3 -- Display the response
        # ---------------------------------------------------------------------------
        # escalate()/decline() never call the LLM, so the placeholder is still
        # blank for those -- st.warning() fills it directly. Everything else
        # overwrites the streamed draft with the final response text, which
        # also covers the case where the Compliance Agent revised the answer
        # after streaming finished (revision itself is never streamed).
        if is_escalated(result):
            placeholder.warning(response)
        else:
            placeholder.markdown(response)
        st.caption(route_label)
        thread["messages"].append({"role": "assistant", "content": response})
        thread["routes"].append(route_label)
        _scroll_to_bottom()


if __name__ == "__main__":
    main()