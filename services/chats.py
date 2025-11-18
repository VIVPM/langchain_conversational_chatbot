import uuid
from services.chat_memory import ensure_memory_from_chat
from utils.time_tools import now_iso

def init_session_state(st):
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("show_signup", False)
    st.session_state.setdefault("show_login", True)
    st.session_state.setdefault("selected_chat_id", None)
    st.session_state.setdefault("chats", {})
    st.session_state.setdefault("memory", None)
    st.session_state.setdefault("is_processing_docs", False)
    st.session_state.setdefault("use_web_search", False)
    st.session_state.setdefault("chat_search_query", "")
    st.session_state.setdefault("chat_displayed_count", 10)
    st.session_state.setdefault("chat_loading_more", False)

def create_new_chat(st) -> str:
    new_chat_id = str(uuid.uuid4())
    ts = now_iso()
    chat = {"id": new_chat_id, "title": "New Chat", "messages": [], "created_at": ts, "updated_at": ts}
    st.session_state.chats[new_chat_id] = chat
    st.session_state.selected_chat_id = new_chat_id
    st.session_state.memory = ensure_memory_from_chat(chat)
    return None

def upsert_and_select_most_recent(st):
    if st.session_state.chats:
        most_recent = max(st.session_state.chats.values(), key=lambda c: c.get("updated_at", c.get("created_at")))
        st.session_state.selected_chat_id = most_recent["id"]

def persist_chats(supabase, user_id, st):
    if supabase:
        supabase.table("profiles").update({"chats": list(st.session_state.chats.values())}).eq("id", user_id).execute()
