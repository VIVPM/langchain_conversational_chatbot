import streamlit as st
from services.chats import create_new_chat
from config import DEFAULT_CHAT_PAGE_SIZE

if "chat_displayed_count" not in st.session_state:
    st.session_state.chat_displayed_count = DEFAULT_CHAT_PAGE_SIZE

def toggle_ws():
    st.session_state.use_web_search = not st.session_state.use_web_search

def render_settings(st):
    st.sidebar.title("Settings")

def render_auth_buttons(st):
    st.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout", key="logout_btn", disabled=st.session_state.is_processing_docs):
        st.session_state.logged_in = False
        st.session_state.chats = {}
        st.session_state.memory = None
        st.session_state.selected_chat_id = None
        st.session_state.show_login = True
        st.session_state.show_signup = False
        st.rerun()

def render_web_search_toggle(st, serper_api_key: str | None):
    enabled = st.session_state.use_web_search and bool(serper_api_key)
    ws_label = f"Web search: {'On' if enabled else 'Off'}"
    new_val = st.sidebar.toggle(ws_label, value=enabled, disabled=not serper_api_key, key="web_search_toggle")
    st.session_state.use_web_search = bool(new_val)
    if not serper_api_key:
        st.sidebar.caption("Add a Serper API key to enable web search.")

def render_chat_search(st):
    st.sidebar.title("Search Chats")
    c1, c2 = st.sidebar.columns([0.85, 0.15])
    with c1:
        q = st.text_input(
            "Search chats",
            value=st.session_state.chat_search_query,
            label_visibility="collapsed",
            placeholder="Search conversations…",
            key="chat_search_input",
            disabled=st.session_state.is_processing_docs,
        )
        if q != st.session_state.chat_search_query:
            st.session_state.chat_search_query = q
            st.session_state.chat_displayed_count = DEFAULT_CHAT_PAGE_SIZE  # reset page
    with c2:
        if st.button("✕", key="chat_search_clear", help="Clear search",
                     disabled=st.session_state.is_processing_docs):
            st.session_state.chat_search_query = ""
            st.session_state.chat_displayed_count = DEFAULT_CHAT_PAGE_SIZE
            st.rerun()

def render_chat_list(st, on_open_chat):
    st.sidebar.title("Chats")
    if st.session_state.is_processing_docs:
        st.sidebar.info("Processing documents. Chats hidden.")
        return

    # "New Chat" button - always enabled
    if st.sidebar.button("New Chat", key="new_chat_btn"):
        # Just clear the selection - don't create anything yet
        st.session_state.selected_chat_id = None
        st.session_state.memory = None
        st.session_state.chat_displayed_count = DEFAULT_CHAT_PAGE_SIZE
        st.rerun()

    # Sort and filter - only show chats that actually have messages
    sorted_chats = sorted(
        [c for c in st.session_state.chats.values() if c.get("messages")],  # ← Filter out empty chats
        key=lambda x: x.get("updated_at", x.get("created_at", "")),
        reverse=True,
    )
    q = st.session_state.chat_search_query.strip().lower()
    if q:
        sorted_chats = [
            c for c in sorted_chats
            if any(q in (m.get("content", "").lower()) for m in c.get("messages", []))
               or q in (c.get("title") or "").lower()
        ]

    # Slice for current page
    start = 0
    end = st.session_state.chat_displayed_count
    displayed = sorted_chats[start:end]

    # Render buttons for displayed chats
    for chat in displayed:
        title = chat.get("title") or "New Chat"
        label = title[:60]
        if st.sidebar.button(label, key=f"open_chat_{chat['id']}"):
            on_open_chat(chat["id"])
            st.rerun()

    # Pager: show more button if more remain
    remaining = max(0, len(sorted_chats) - end)
    if remaining > 0:
        show_n = min(remaining, DEFAULT_CHAT_PAGE_SIZE)
        if st.sidebar.button(f"Show more ({show_n})", key="show_more_chats"):
            st.session_state.chat_displayed_count += DEFAULT_CHAT_PAGE_SIZE
            st.rerun()
    else:
        if sorted_chats:
            st.sidebar.caption("No more chats.")