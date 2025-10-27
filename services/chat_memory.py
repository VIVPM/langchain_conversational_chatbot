from langchain.memory import ConversationBufferMemory  # This should work with langchain-community installed
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

def ensure_memory_from_chat(chat) -> ConversationBufferMemory:
    # Use ChatMessageHistory explicitly
    chat_history = ChatMessageHistory()
    
    for m in chat.get("messages", []):
        if m["role"] == "user":
            chat_history.add_message(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.add_message(AIMessage(content=m["content"]))
    
    mem = ConversationBufferMemory(
        chat_memory=chat_history,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return mem

def render_history_text(mem: ConversationBufferMemory) -> str:
    msgs = mem.load_memory_variables({}).get("chat_history", [])
    parts = []
    for m in msgs:
        t = getattr(m, "type", None) or m.__class__.__name__.lower()
        if t in ("human", "humanmessage"):
            parts.append(f"User: {m.content}")
        elif t in ("ai", "aimessage"):
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(m.content)
    return "\n".join(parts)