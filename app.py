import streamlit as st
from config import MAX_FILES, MAX_FILE_SIZE_BYTES, DEFAULT_MODEL
from services.supabase_client import get_supabase
from services.chats import init_session_state, create_new_chat, persist_chats
from services.chat_memory import ensure_memory_from_chat, render_history_text
from services.new_vectorstore import make_vectorstore
from services.ingest import split_files, index_docs
from services.llm import get_llm, answer_direct, answer_from_context
from services.search import web_search_answer
from services.auth import signup, login
from ui.sidebar import render_settings, render_web_search_toggle, render_auth_buttons, render_chat_search, render_chat_list
from ui.chat_window import render_chat_history
from utils.ratelimit import allow
from utils.time_tools import now_iso
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

st.title("QuickBrain")

# state
init_session_state(st)
supabase = get_supabase()

# Auth screens
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        st.subheader("Signup")
        with st.form("signup_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submit = st.form_submit_button("Signup")
        if submit:
            err = signup(supabase, u, p, st)
            if err: st.error(err)
            else: st.rerun()
        if st.button("Login", key="login_acct_btn"):
            st.session_state.show_login, st.session_state.show_signup = True, False
            st.rerun()
        st.stop()

    if st.session_state.show_login:
        st.subheader("Login")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
        if submit:
            err = login(supabase, u, p, st)
            if err: st.error(err)
            else: st.rerun()
        if st.button("Create new account", key="create_acct_btn"):
            st.session_state.show_login, st.session_state.show_signup = False, True
            st.rerun()
        st.stop()

# Main app
render_settings(st)
api_key = st.sidebar.text_input("SambaNova API Key", type="password", disabled=st.session_state.is_processing_docs)
model_choice = st.sidebar.selectbox("Choose model", options=[
    "Llama-3.3-Swallow-70B-Instruct-v0.4",
    "Llama-4-Maverick-17B-128E-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.3-70B-Instruct"
], index=3, disabled=st.session_state.is_processing_docs)
serper_api_key = st.sidebar.text_input("Serper API Key", type="password", disabled=st.session_state.is_processing_docs)
render_web_search_toggle(st, serper_api_key)
render_auth_buttons(st)

# Upload + process
uploaded_files = st.file_uploader("Upload up to 3 files each under 500KB", type=["pdf","txt","docx"], accept_multiple_files=True, disabled=st.session_state.is_processing_docs) or []
if len(uploaded_files) > MAX_FILES:
    st.error("Select at most 3 Files.")
    st.stop()
oversized = [uf for uf in uploaded_files if getattr(uf, "size", 0) > MAX_FILE_SIZE_BYTES]
if oversized:
    st.error("Each file must be under 500KB: " + ", ".join(f.name for f in oversized))
else:
    if st.button("Process Documents", disabled=st.session_state.is_processing_docs or not (uploaded_files and api_key)):
        st.session_state.is_processing_docs = True
        with st.spinner("Processing documents..."):
            vs = make_vectorstore(st.session_state.username)
            docs = split_files(uploaded_files)
            n = index_docs(vs, docs)
            st.session_state.vectorstore = vs
        st.session_state.is_processing_docs = False
        st.success(f"Indexed {n} chunks.")
        st.rerun()

# Sidebar: search and chat list
render_chat_search(st)
def on_open_chat(chat_id: str):
    st.session_state.selected_chat_id = chat_id
    st.session_state.memory = ensure_memory_from_chat(st.session_state.chats[chat_id])
render_chat_list(st, on_open_chat)

# Chat window
if st.session_state.selected_chat_id:
    current_chat = st.session_state.chats[st.session_state.selected_chat_id]
    if st.session_state.memory is None:
        st.session_state.memory = ensure_memory_from_chat(current_chat)
    render_chat_history(current_chat)
else:
    st.info("Select a chat or start a new one.")

# Input + LLM
if api_key and st.session_state.selected_chat_id and not st.session_state.is_processing_docs:
    q = st.chat_input("Your question")
    if q:
        cur = st.session_state.chats[st.session_state.selected_chat_id]
        t = now_iso()
        if cur["title"] == "New Chat":
            cur["title"] = q[:50] + ("..." if len(q) > 50 else "")
            st.session_state["_sidebar_title_just_updated"] = True

        cur["messages"].append({"role": "user", "content": q, "date": t})
        with st.chat_message("user"): st.markdown(q)

        with st.spinner("Generating answer..."):
            if not allow("llm_calls", limit=50, window_sec=180):
                st.warning("Rate limit: 50 requests per minute.")
                st.stop()
            llm = get_llm(model_choice or DEFAULT_MODEL, api_key)
            history_text = render_history_text(st.session_state.memory)

            if st.session_state.use_web_search and serper_api_key:
                final, sources = web_search_answer(llm, serper_api_key, q)
                source_block = "\n\n**Web Sources used:**\n" + "\n".join(f"- {s}" for s in sources) if sources else ""
            else:
                final, source_block = None, ""
                if "vectorstore" in st.session_state:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10}, search_type="mmr")
                    docs = retriever.get_relevant_documents(q)
                    context = "\n\n".join(d.page_content for d in docs)
                    rag = answer_from_context(llm, context, q)
                    if rag and "Information not found in the provided documents." not in rag:
                        final = rag
                        uniq_src = sorted({d.metadata.get("source","") for d in docs if d.metadata.get("source")})
                        if uniq_src:
                            source_block = "\n\n**Sources used:**\n" + "\n".join(f"- {s}" for s in uniq_src)
                if not final:
                    final = answer_direct(llm, history_text, q)

        assistant_response = f"**Answer:**\n{final}{source_block}"
        cur["messages"].append({"role": "assistant", "content": assistant_response, "date": t})
        cur["updated_at"] = now_iso()
        st.session_state.memory.chat_memory.add_user_message(q)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        with st.chat_message("assistant"): st.markdown(assistant_response)

        persist_chats(supabase, st.session_state.user_id, st)
        if st.session_state.pop("_sidebar_title_just_updated", False):
            st.rerun()
else:
    if not api_key:
        st.info("Enter your SambaNova API key in the sidebar.")
    elif st.session_state.selected_chat_id is None:
        st.info("Select a chat from the sidebar or create a new one.")
