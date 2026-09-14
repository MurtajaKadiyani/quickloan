"""
app.py
------
Streamlit chat UI for QuickLoan -- FastFinance's AI loan assistant.

Session 14: Security and Guardrails.
  - Guard node runs before every query -- blocks injection and PII, and
    (via _llamaguard_safe()) semantic jailbreaks, before any LLM is called
  - Supervisor classifies clean queries (RATES / POLICY / RATES+POLICY /
    COMPLEX / OUT_OF_SCOPE)
  - Rates Agent (MCP) / Policy Agent (RAG) / call_both_agents (compound
    queries) handle them
  - Compliance Agent checks every draft response
  - Human-in-the-Loop: operator must approve compliance-revised responses
    before they reach the customer (_handle_hitl())

Also carries forward from earlier sessions (not part of the S14 guard work,
but already built out beyond the bare starter template):
  - Multi-conversation sidebar (_new_thread/_clear_current_thread/_delete_thread)
    -- each browser session can hold several independent threads against the
    same shared MemorySaver checkpointer
  - Char-by-char streaming with a blinking cursor (_StreamingState/_CURSOR_CSS)
  - Auto-scroll to the latest message on every turn (_scroll_to_bottom())

Run:
    streamlit run app.py   (from inside s01/starter/)
"""
import json
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
# build_input_state() -- the graph's initial state for graph.invoke()
# ---------------------------------------------------------------------------
# blocked_reason/llamaguard_score (S14) reset the Input Guard fields each turn
# for the same reason the other fields are reset -- QuickLoanState is
# checkpointed per thread_id, so without an explicit reset here a stale value
# from a previous turn would otherwise carry forward into this one.
def build_input_state(message: str) -> dict:
    return {
        "customer_message":  message,
        "response":          "",
        "specialist":        "",
        "retrieved_docs":    [],
        "compliance_status": "",
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
    }


def get_thread_config(thread_id: str) -> dict:
    """LangGraph needs a thread ID to keep memory across turns in the same session."""
    return {"configurable": {"thread_id": thread_id}}


def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ RBI Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def guard_badge(blocked_reason: str, llamaguard_score: float = -1.0) -> str:
    """S14: return a guard status badge, including the LlamaPromptGuard score
    when Layer 2 (semantic) is what caught it -- Layer 1 (regex, PII/injection)
    never reaches the guard model, so there's no score to show for those."""
    if blocked_reason == "pii":
        return "🔒 Blocked (PII)"
    if blocked_reason == "llamaguard":
        score_str = f" · score {llamaguard_score:.4f}" if llamaguard_score >= 0 else ""
        return f"🤖 Blocked (jailbreak — Prompt Guard{score_str})"
    if blocked_reason:
        return "🛡️ Blocked (injection — regex)"
    return ""


def needs_human_review(result: dict) -> bool:
    """S14: True when the Compliance Agent revised the response -- the operator
    must approve (or discard) the rewrite before it reaches the customer."""
    return result.get("compliance_status", "") == "REVISED"


def format_route_label(result: dict) -> str:
    blocked_r = result.get("blocked_reason", "")
    if blocked_r:
        return f"Guard: {guard_badge(blocked_r, result.get('llamaguard_score', -1.0))}"

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


def _search_thread_ids(threads: dict, query: str) -> list:
    """Return thread_ids (most-recently-created first) whose title or any
    message content contains query, case-insensitive. Empty query matches
    every thread -- same as no search being active."""
    tids = list(reversed(list(threads.keys())))
    q = query.strip().lower()
    if not q:
        return tids
    matches = []
    for tid in tids:
        thread = threads[tid]
        if q in thread.get("title", "").lower():
            matches.append(tid)
            continue
        if any(q in msg.get("content", "").lower() for msg in thread.get("messages", [])):
            matches.append(tid)
    return matches


def _thread_match_is_title_only(thread: dict, query: str) -> bool:
    """True if query matched the title -- used to decide whether to show a
    '(message match)' hint for threads found by content instead of title."""
    q = query.strip().lower()
    return q in thread.get("title", "").lower()


def _inject_live_search_filter(threads: dict) -> None:
    """Filter the sidebar's conversation rows on every keystroke, not just
    on Enter/blur (st.text_input only reruns Streamlit -- and so only
    updates search_query -- on those two events, not per character).

    Reaches into window.parent.document the same way _scroll_to_bottom()
    does (components.html() renders in a same-origin iframe; there's no
    other channel to push a live value back into Python without a full
    custom component). Each conversation row is `key=f"thread_btn_{tid}"`,
    which Streamlit documents as producing a `st-key-thread_btn_{tid}` CSS
    class -- used here only to find the row's [data-testid="stHorizontalBlock"]
    ancestor to hide/show, never to read or execute anything from it. If a
    future Streamlit version changes this convention, every DOM lookup below
    is null-guarded so the script quietly does nothing -- the CSS-only
    filtering in _sidebar() (driven by the committed search value) still
    works as the reliable fallback regardless.

    Security note: thread titles and message content are real customer/LLM
    text, not app-authored strings, so they cannot be interpolated into this
    <script> block as raw text -- a message containing a literal "</script>"
    sequence would otherwise terminate the block early (this is the same
    class of risk the rest of the app avoids by never using
    unsafe_allow_html=True on LLM output). json.dumps() escapes quotes/
    control characters correctly for a JS string/object literal, but not
    "</script>" specifically (JSON has no concept of HTML/script context),
    so "<" is additionally escaped to the unicode form -- the resulting text
    is only ever used for a JS .includes() substring check, never rendered
    as HTML or eval'd, so this one substitution is sufficient.

    Title and message content are kept as separate fields (not one combined
    string) so this script can also live-update the "(message match)" hint
    -- _sidebar()'s Python-side _thread_match_is_title_only() only recomputes
    that hint when search_query is committed (Enter/blur), so a purely
    visual show/hide here left the hint frozen at whatever it was on the
    last commit while the rows themselves already filtered live -- verified
    2026-09-12: typing a message-only match with no Enter narrowed the list
    correctly but never showed "(message match)" until Enter was pressed.
    """
    search_index = {
        tid: {
            "title": thread.get("title", ""),
            "content": " ".join(msg.get("content", "") for msg in thread.get("messages", [])),
        }
        for tid, thread in threads.items()
    }
    search_index_json = json.dumps(search_index).replace("<", "\\u003c")
    match_suffix = "  (message match)"

    components.html(
        f"""
        <script>
            (function() {{
                var doc = window.parent.document;
                var index = {search_index_json};
                var suffix = {json.dumps(match_suffix)};
                var input = doc.querySelector('input[aria-label="🔍 Search conversations"]');
                var noMatch = doc.getElementById('ql-no-match-msg');
                if (!input) return;

                function applyFilter() {{
                    var q = (input.value || '').trim().toLowerCase();
                    var anyVisible = false;
                    for (var tid in index) {{
                        if (!Object.prototype.hasOwnProperty.call(index, tid)) continue;
                        var btn = doc.querySelector('.st-key-thread_btn_' + tid);
                        var row = btn ? btn.closest('[data-testid="stHorizontalBlock"]') : null;
                        if (!row) continue;

                        var title = index[tid].title.toLowerCase();
                        var content = index[tid].content.toLowerCase();
                        var titleMatches = !q || title.indexOf(q) !== -1;
                        var anyMatch = titleMatches || content.indexOf(q) !== -1;
                        row.style.display = anyMatch ? '' : 'none';
                        if (anyMatch) anyVisible = true;

                        // Live-update the "(message match)" hint -- strip any
                        // stale suffix (server-rendered from the last commit,
                        // or added by an earlier keystroke) then re-add it
                        // only when this row is showing purely because of a
                        // message-content match, not the title.
                        var labelBtn = btn.querySelector('button');
                        if (labelBtn) {{
                            var base = labelBtn.textContent.endsWith(suffix)
                                ? labelBtn.textContent.slice(0, -suffix.length)
                                : labelBtn.textContent;
                            var needsSuffix = q && anyMatch && !titleMatches;
                            var desired = needsSuffix ? base + suffix : base;
                            if (labelBtn.textContent !== desired) {{
                                labelBtn.textContent = desired;
                            }}
                        }}
                    }}
                    if (noMatch) {{
                        noMatch.style.display = (q && !anyVisible) ? 'block' : 'none';
                    }}
                }}

                input.addEventListener('input', applyFilter);
                applyFilter();
            }})();
        </script>
        """,
        height=0,
    )


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
        search_query = st.text_input(
            "🔍 Search conversations",
            key="chat_search",
            placeholder="Search by title or message…",
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Conversations")

        # Every thread is always rendered (buttons stay live/clickable) --
        # filtering is purely visual, layered two ways:
        #   1. CSS, driven by search_query (the last *committed* value --
        #      text_input only updates on Enter or losing focus, not every
        #      keystroke). Reliable, no JS required; this is what applies
        #      right after a real rerun.
        #   2. A live per-keystroke JS filter injected below
        #      (_inject_live_search_filter), so the list narrows instantly
        #      while typing instead of waiting for Enter/blur. If it ever
        #      breaks (e.g. a future Streamlit version changes internal
        #      testids), layer 1 still works -- typing + Enter/clicking away
        #      always filters correctly regardless.
        all_tids      = list(reversed(list(st.session_state.threads.keys())))
        matching_tids = _search_thread_ids(st.session_state.threads, search_query)
        hidden_tids   = [tid for tid in all_tids if tid not in matching_tids] if search_query else []

        no_match_display = "block" if (search_query and not matching_tids) else "none"
        st.markdown(
            f'<p id="ql-no-match-msg" style="display:{no_match_display}; '
            'color: var(--text-color, #888); opacity: 0.6; font-size: 0.85rem;">'
            "No conversations match your search.</p>",
            unsafe_allow_html=True,
        )
        if hidden_tids:
            hide_css = "\n".join(
                f'div[data-testid="stHorizontalBlock"]:has(.st-key-thread_btn_{tid}) '
                "{ display: none !important; }"
                for tid in hidden_tids
            )
            st.markdown(f"<style>{hide_css}</style>", unsafe_allow_html=True)

        for tid in all_tids:
            thread = st.session_state.threads[tid]
            active = tid == st.session_state.thread_id
            label  = ("🟢 " if active else "💬 ") + thread["title"]
            if search_query and tid in matching_tids and not _thread_match_is_title_only(thread, search_query):
                label += "  (message match)"
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

        _inject_live_search_filter(st.session_state.threads)

        st.caption(f"Session: {st.session_state.thread_id[:8]}…")
        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Guard** — blocks injections & PII *(S14)*\n"
            "- **Supervisor** — classifies query\n"
            "- **Rates Agent** — live interest rates via MCP\n"
            "- **Policy Agent** — loan policy via RAG\n"
            "- **call_both_agents** — Rates + Policy together, for compound queries\n"
            "- **Compliance Agent** — RBI rules check\n"
            "- **Human-in-the-Loop** — reviews revisions *(S14)*\n"
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

    Streamlit's own chat layout auto-follows the bottom already *while the
    user is already there*, but does NOT force-scroll them back down if
    they've scrolled up to reread an earlier turn -- verified live
    (2026-09-12): scrollTop stayed pinned at 0 for the whole reply, leaving
    the user ~650px above newly-arrived content with no indication anything
    happened. That's the exact case this function exists for.

    `section.main` (the original selector here) no longer exists in current
    Streamlit versions -- confirmed via direct DOM inspection, the actual
    scrollable element is now `[data-testid="stAppScrollToBottomContainer"]`
    (a wrapper Streamlit itself added for native bottom-following). The old
    selector silently fell through to the `|| doc.body` fallback, which
    isn't the real scroll container either (body's scrollHeight reads 0
    here), so this function has been a no-op for a while without erroring --
    the app *looked* fine because Streamlit's own follow-when-already-at-
    bottom behavior covered the common case, just not this one. Both
    testids are still queried (old one first) in case a future Streamlit
    version renames the container again; if neither matches, this quietly
    does nothing rather than throwing, same as before.

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
            var scroller = doc.querySelector('section.main')
                || doc.querySelector('[data-testid="stAppScrollToBottomContainer"]')
                || doc.body;
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


def _sync_hitl_decision(thread_id: str, final_text) -> None:
    """Reconcile the graph's checkpointed history with what the operator
    actually decided in the HITL approval form.

    call_compliance_agent() (nodes.py) already overwrites the checkpointed
    history's last assistant turn with revise_response()'s text *before*
    graph.invoke() even returns here -- that happens unconditionally, ahead
    of any operator review. If the operator then edits the text before
    approving, or discards it outright, the checkpoint silently keeps
    disagreeing with what the customer actually saw (or never saw), and that
    stale turn still feeds into the model's own context on the customer's
    next message -- e.g. the model could "remember" saying something the
    customer never received, or a corrected number the operator overrode.

    final_text=None means the turn was discarded entirely -- remove both the
    user question and the unsent assistant answer, since neither reached the
    visible conversation. Otherwise replace the assistant turn with
    final_text (the possibly-edited, approved text) so the checkpoint matches
    reality.
    """
    config = get_thread_config(thread_id)
    try:
        snapshot = st.session_state.graph.get_state(config)
        history  = list(snapshot.values.get("history", []))
    except Exception as e:
        print(f"[QuickLoan] Could not read checkpointed history for {thread_id}: {e}")
        return

    if not history or history[-1].get("role") != "assistant":
        return  # nothing to reconcile

    if final_text is None:
        history = history[:-1]
        if history and history[-1].get("role") == "user":
            history = history[:-1]
    else:
        history = history[:-1] + [{"role": "assistant", "content": final_text}]

    try:
        st.session_state.graph.update_state(config, {"history": history})
    except Exception as e:
        print(f"[QuickLoan] Could not sync checkpointed history for {thread_id}: {e}")


def _handle_hitl() -> bool:
    """S14: render the compliance-revision approval form if one is pending.

    Returns True while a decision is outstanding -- callers should skip
    rendering st.chat_input() in that case, since the conversation can't
    advance until the operator approves or discards the revised draft.

    The target thread_id is captured inside pending_hitl at the moment the
    revision happened (not "whichever thread is active now") so that
    switching threads in the sidebar while a review is pending still resolves
    to the right conversation instead of appending into whatever the operator
    happens to be looking at when they click Approve/Discard.
    """
    pending = st.session_state.get("pending_hitl")
    if pending is None:
        return False

    st.warning(
        "⚠️ **Compliance Review Required** — The Compliance Agent revised this response. "
        "Please review and approve before sending to the customer."
    )

    with st.form("hitl_approval"):
        edited = st.text_area(
            "Review and edit the response if needed:",
            value=pending["response"],
            height=220,
        )
        col1, col2 = st.columns(2)
        approved  = col1.form_submit_button("✅ Approve & Send", use_container_width=True)
        discarded = col2.form_submit_button("❌ Discard",         use_container_width=True)

    if approved:
        thread = st.session_state.threads.get(pending["thread_id"])
        if thread is not None:
            thread["messages"].append({"role": "assistant", "content": edited})
            thread["routes"].append(pending["route_label"])
        _sync_hitl_decision(pending["thread_id"], edited)
        st.session_state.pop("pending_hitl", None)
        _scroll_to_bottom()
        st.rerun()
    elif discarded:
        _sync_hitl_decision(pending["thread_id"], None)
        st.session_state.pop("pending_hitl", None)
        st.rerun()

    return True


def main() -> None:
    st.set_page_config(page_title="QuickLoan | FastFinance", page_icon="💰", layout="wide")
    st.title("💰 QuickLoan | FastFinance")
    st.caption("AI-powered loan assistant — Session 14: Security and Guardrails")
    st.markdown(_CURSOR_CSS, unsafe_allow_html=True)

    _init_session()
    _sidebar()
    _render_history()

    hitl_active = _handle_hitl()

    if not hitl_active:
        prompt = st.chat_input("Ask about loan rates, eligibility, or our policies…")
        if prompt:
            thread = st.session_state.threads[st.session_state.thread_id]
            # _sidebar() (which renders the conversation list) already ran
            # earlier in this same script pass -- setting thread["title"]
            # here updates the dict, but the sidebar UI already emitted its
            # old "New chat" label and Streamlit doesn't retroactively
            # re-render it within one run. Without an extra rerun once the
            # response finishes, the sidebar shows "New chat" for this
            # thread until the *next* unrelated interaction happens to
            # trigger one. Remember it's a new thread here; each response
            # branch below reruns once at the end only when this is True, so
            # ongoing turns in an existing thread never pay for an extra
            # rerun (which would also blank the streamed text and replace it
            # with a fresh render).
            is_first_message_in_thread = not thread["messages"]
            if is_first_message_in_thread:
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
            # nodes.py's _generate_response_text() streams the reply. The
            # Guard node runs before classify()/any specialist, so a blocked
            # message never streams anything into it -- it stays blank until
            # the blocked_r branch below fills it directly.
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
            blocked_r   = result.get("blocked_reason", "")

            # escalate()/decline() and a Guard block never call the LLM, so the
            # placeholder is still blank for those -- st.warning() fills it
            # directly. A compliance revision hands off to _handle_hitl()
            # instead of displaying here: the streamed draft in the placeholder
            # is the PRE-revision text (compliance runs after respond()), so it
            # must not reach the customer -- clear it and wait for operator
            # approval. Everything else overwrites the streamed draft with the
            # final response text.
            if blocked_r:
                placeholder.warning(response)
                st.caption(route_label)
                thread["messages"].append({"role": "assistant", "content": response})
                thread["routes"].append(route_label)
                _scroll_to_bottom()
                if is_first_message_in_thread:
                    st.rerun()
            elif needs_human_review(result):
                placeholder.empty()
                st.session_state.pending_hitl = {
                    "response":    response,
                    "route_label": route_label,
                    "thread_id":   st.session_state.thread_id,
                }
                st.rerun()
            else:
                if is_escalated(result):
                    placeholder.warning(response)
                else:
                    # Compound RATES+POLICY queries never stream live -- see
                    # nodes.py's call_both_agents(), which deliberately sets
                    # _stream_callback to None for the duration of its two
                    # concurrent sub-agent calls (both share the module-level
                    # `llm` client; two simultaneous llm.stream() reads on it
                    # corrupt each other). That's the right call for the LLM
                    # calls themselves, but it left the placeholder blank
                    # through the whole generation and this file used to just
                    # dump the finished text in with one instant markdown()
                    # -- the "Typing speed" setting had no effect at all for
                    # this one route. Fixed by replaying the now-complete
                    # merged text through the same typewriter mechanism as a
                    # single post-hoc pass: no llm.stream()/threads involved
                    # here, so none of the concurrency risk above applies.
                    if result.get("specialist") == "rates_agent+policy_agent":
                        _StreamingState(placeholder, char_delay=char_delay)(response)
                    placeholder.markdown(response)
                st.caption(route_label)
                thread["messages"].append({"role": "assistant", "content": response})
                thread["routes"].append(route_label)
                _scroll_to_bottom()
                if is_first_message_in_thread:
                    st.rerun()


if __name__ == "__main__":
    main()