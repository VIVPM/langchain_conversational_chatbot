import streamlit as st

def render_chat_history(current_chat):
    for message in current_chat.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
