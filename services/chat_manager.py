import uuid
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import SambaNovaCloud
from config import DEFAULT_MODEL
from utils.common_utils import now_iso


class HybridMemory:
    """
    Hybrid memory that keeps recent messages verbatim and summarizes old ones.
    
    - window_size: Number of recent messages to keep verbatim
    - summarize_threshold: Total messages before triggering summarization
    """
    
    def __init__(self, llm=None, window_size=6, summarize_threshold=15):
        self.llm = llm
        self.window_size = window_size
        self.summarize_threshold = summarize_threshold
        self.messages = []
        self.summary = ""
    
    def add_user_message(self, content: str):
        """Add a user message to memory."""
        self.messages.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """Add an AI message and check if summarization needed."""
        self.messages.append(AIMessage(content=content))
        self._maybe_summarize()
    
    def _maybe_summarize(self):
        """Summarize old messages when threshold exceeded."""
        if len(self.messages) <= self.summarize_threshold:
            return
        
        if not self.llm:
            # No LLM available, just truncate
            self.messages = self.messages[-self.window_size:]
            return
        
        try:
            # Messages to summarize (all except recent window)
            to_summarize = self.messages[:-self.window_size]
            
            # Build text for summarization
            text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in to_summarize
            )
            
            # Include existing summary for continuity
            if self.summary:
                text = f"Previous summary: {self.summary}\n\nNew messages:\n{text}"
            
            # Summarize with LLM (ONE call)
            prompt = f"""Summarize this conversation concisely, keeping key facts, names, and decisions:

                    {text}

                    Summary:"""
            self.summary = self.llm.invoke(prompt).strip()
            
            # Keep only recent messages
            self.messages = self.messages[-self.window_size:]
        except ConnectionError:
            # Network/API connection failed - truncate and continue
            self.messages = self.messages[-self.window_size:]
        except TimeoutError:
            # LLM took too long - truncate and continue
            self.messages = self.messages[-self.window_size:]
        except ValueError as e:
            # Invalid response from LLM - truncate and continue
            self.messages = self.messages[-self.window_size:]
        except Exception as e:
            # Unexpected error - log type for debugging, truncate and continue
            print(f"HybridMemory summarization failed ({type(e).__name__}): {str(e)[:50]}")
            self.messages = self.messages[-self.window_size:]
    
    def get_context(self) -> str:
        """Get full context for LLM prompt."""
        parts = []
        
        if self.summary:
            parts.append(f"[Previous conversation summary]\n{self.summary}\n")
        
        if self.messages:
            parts.append("[Recent messages]")
            for m in self.messages:
                role = "User" if isinstance(m, HumanMessage) else "Assistant"
                parts.append(f"{role}: {m.content}")
        
        return "\n".join(parts)
    
    def save_context(self, inputs: dict, outputs: dict):
        """Compatibility method for old code that uses save_context."""
        if "input" in inputs:
            self.add_user_message(inputs["input"])
        if "answer" in outputs:
            self.add_ai_message(outputs["answer"])


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


def ensure_memory_from_chat(chat, llm=None) -> HybridMemory:
    """Create HybridMemory and load existing messages from chat."""
    # If no LLM provided, try to create one from env
    if llm is None:
        api_key = os.getenv('SAMBANOVA_API_KEY')
        if api_key:
            try:
                llm = SambaNovaCloud(model=DEFAULT_MODEL, sambanova_api_key=api_key, max_tokens=500, temperature=0.3)
            except Exception:
                llm = None
    
    mem = HybridMemory(llm=llm, window_size=6, summarize_threshold=15)
    
    for m in chat.get("messages", []):
        if m["role"] == "user":
            mem.add_user_message(m["content"])
        elif m["role"] == "assistant":
            mem.add_ai_message(m["content"])
    
    return mem


def create_new_chat(st) -> str:
    new_chat_id = str(uuid.uuid4())
    ts = now_iso()
    chat = {"id": new_chat_id, "title": "New Chat", "messages": [], "created_at": ts, "updated_at": ts}
    st.session_state.chats[new_chat_id] = chat
    st.session_state.selected_chat_id = new_chat_id
    st.session_state.memory = ensure_memory_from_chat(chat)
    return None


def persist_chats(supabase, user_id, st):
    if supabase:
        try:
            supabase.table("profiles").update({"chats": list(st.session_state.chats.values())}).eq("id", user_id).execute()
        except Exception as e:
            st.warning(f"Failed to save chat: {str(e)[:50]}. Your messages may not be saved.")


def render_history_text(mem) -> str:
    """Compatibility function - now just calls mem.get_context()."""
    if mem is None:
        return ""
    if hasattr(mem, 'get_context'):
        return mem.get_context()
