import time
from collections import deque
import streamlit as st

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
