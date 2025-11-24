import hashlib
import time
from collections import deque
from datetime import datetime
import pytz
import streamlit as st
from config import TIMEZONE
from docx import Document

# --- Crypto ---
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def chunk_id(text: str, source: str) -> str:
    return hashlib.sha1((text + source).encode()).hexdigest()

# --- Files ---
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type")

# --- Rate Limit ---
def allow(action: str, limit: int, window_sec: int) -> bool:
    key = f"rl_{action}"
    q = st.session_state.setdefault(key, deque())
    now = time.time()
    while q and now - q[0] > window_sec:
        q.popleft()
    if len(q) < limit:
        q.append(now)
        return True
    return False

# --- Time Tools ---
def now_iso() -> str:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).isoformat(timespec="seconds")
