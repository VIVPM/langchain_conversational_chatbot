import re
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
from utils.common_utils import sha256, now_iso

def get_supabase() -> Client | None:
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

def valid_username(u: str) -> bool:
    return bool(re.fullmatch(r"^[A-Za-z0-9._%+-]+@gmail\.com$", u))

def signup(supabase, username: str, password: str, st):
    if not valid_username(username):
        return "Username must be a valid address ending with Gmail Account."
    if not supabase:
        return "Supabase not configured."
    if supabase.table("profiles").select("*").eq("username", username).execute().data:
        return "Username already exists."
    password_hash = sha256(password)
    resp = supabase.table("profiles").insert(
        {"username": username, "password": password_hash, "chats": []}
    ).execute()
    
    user_row = (resp.data or supabase.table("profiles").select("*").eq("username", username).execute().data)[0]
    st.session_state.logged_in = True
    st.session_state.user_id = user_row["id"]
    st.session_state.username = username
    st.session_state.chats = {}
    st.session_state.selected_chat_id = None
    st.session_state.memory = None
    return None

def login(supabase, username: str, password: str, st):
    if not supabase:
        return "Supabase not configured."
    user_data = supabase.table("profiles").select("*").eq("username", username).execute().data
    if not user_data:
         return "Invalid username or password."
    user = user_data[0]
    
    if not user['username'] or sha256(password) != user["password"]:
        return "Invalid username or password."

    st.session_state.logged_in = True
    st.session_state.user_id = user["id"]
    st.session_state.username = username

    chats = user.get("chats", []) or []
    for c in chats:
        c.setdefault("created_at", c.get("date", now_iso()))
        c.setdefault("updated_at", c.get("created_at"))
    st.session_state.chats = {c["id"]: c for c in chats}

    # Clean up multiple empty "New Chat" entries if they exist
    empty_news = [
        c for c in st.session_state.chats.values()
        if (c.get("title") == "New Chat") and not c.get("messages")
    ]

    if len(empty_news) > 1:
        # Keep the most recent, remove others
        target = max(empty_news, key=lambda c: c.get("updated_at", c.get("created_at")))
        to_drop = [c for c in empty_news if c["id"] != target["id"]]
        for c in to_drop:
            st.session_state.chats.pop(c["id"], None)
        supabase.table("profiles").update(
            {"chats": list(st.session_state.chats.values())}
        ).eq("id", st.session_state.user_id).execute()
    
    # Don't auto-select any chat - let user choose or start typing
    st.session_state.selected_chat_id = None
    st.session_state.memory = None
    
    return None
