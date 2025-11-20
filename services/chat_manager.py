import uuid
import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms import SambaNovaCloud
from config import DEFAULT_MODEL
from utils.common_utils import now_iso

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

def ensure_memory_from_chat(chat) -> ConversationSummaryBufferMemory:
    mem = ConversationSummaryBufferMemory(
        llm = SambaNovaCloud(model=DEFAULT_MODEL,sambanova_api_key=os.getenv('SAMBANOVA_API_KEY'),max_tokens=1000,temperature=0.4),
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=1000,
        input_key="input",    # ← Explicit
        output_key="answer"   # ← Explicit
    )
    for m in chat.get("messages", []):
        if m["role"] == "user":
            mem.chat_memory.add_user_message(m["content"])
        elif m["role"] == "assistant":
            mem.chat_memory.add_ai_message(m["content"])
    
    return mem

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

def render_history_text(mem: ConversationSummaryBufferMemory) -> str:
    msgs = mem.load_memory_variables({}).get("chat_history", [])
    parts = []
    for m in msgs:
        t = getattr(m, "type", None) or m.__class__.__name__.lower()
        if t in ("system", "systemmessage"):
            parts.append(f"Summary of previous chat messages: {m.content}")
        elif t in ("human", "humanmessage"):
            parts.append(f"User: {m.content}")
        elif t in ("ai", "aimessage"):
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(m.content)
    return "\n".join(parts)
