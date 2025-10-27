import re
from utils.crypto import sha256
from services.chats import create_new_chat
from utils.time_tools import now_iso

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
    # fetch created row
    user_row = (resp.data or supabase.table("profiles").select("*").eq("username", username).execute().data)[0]
    st.session_state.logged_in = True
    st.session_state.user_id = user_row["id"]
    st.session_state.username = username
    st.session_state.chats = {}
    create_new_chat(st)
    # persist
    supabase.table("profiles").update({"chats": list(st.session_state.chats.values())}).eq("id", st.session_state.user_id).execute()
    return None

def login(supabase, username: str, password: str, st):
    if not supabase:
        return "Supabase not configured."
    user = supabase.table("profiles").select("*").eq("username", username).execute()
    if not user.data or sha256(password) != user.data[0]["password"]:
        return "Invalid username or password."

    st.session_state.logged_in = True
    st.session_state.user_id = user.data[0]["id"]
    st.session_state.username = username

    chats = user.data[0].get("chats", []) or []
    for c in chats:
        c.setdefault("created_at", c.get("date", now_iso()))
        c.setdefault("updated_at", c.get("created_at"))
    st.session_state.chats = {c["id"]: c for c in chats}

    # 1) find any empty "New Chat"
    empty_news = [
        c for c in st.session_state.chats.values()
        if (c.get("title") == "New Chat") and not c.get("messages")
    ]

    if empty_news:
        # 2) select the most recent empty "New Chat"
        target = max(empty_news, key=lambda c: c.get("updated_at", c.get("created_at")))
        st.session_state.selected_chat_id = target["id"]

        # 3) if there are multiple empties, keep one, drop the rest
        to_drop = [c for c in empty_news if c["id"] != target["id"]]
        for c in to_drop:
            st.session_state.chats.pop(c["id"], None)
        if to_drop:
            supabase.table("profiles").update(
                {"chats": list(st.session_state.chats.values())}
            ).eq("id", st.session_state.user_id).execute()
    else:
        # 4) none exists → create exactly one
        new_id = create_new_chat(st)
        st.session_state.selected_chat_id = new_id
        supabase.table("profiles").update(
            {"chats": list(st.session_state.chats.values())}
        ).eq("id", st.session_state.user_id).execute()

    return None
