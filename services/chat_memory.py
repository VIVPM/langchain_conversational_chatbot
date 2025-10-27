from langchain.memory import ConversationBufferMemory

def ensure_memory_from_chat(chat) -> ConversationBufferMemory:
    mem = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    for m in chat.get("messages", []):
        if m["role"] == "user":
            mem.chat_memory.add_user_message(m["content"])
        elif m["role"] == "assistant":
            mem.chat_memory.add_ai_message(m["content"])
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
