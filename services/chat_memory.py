from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

def ensure_memory_from_chat(chat):
    """Modern replacement using ChatMessageHistory"""
    history = ChatMessageHistory()
    
    for m in chat.get("messages", []):
        if m["role"] == "user":
            history.add_message(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.add_message(AIMessage(content=m["content"]))
    
    return history

def render_history_text(history: ChatMessageHistory) -> str:
    """Render chat history as text"""
    parts = []
    for m in history.messages:
        if isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    return "\n".join(parts)