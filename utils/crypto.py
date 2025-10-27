import hashlib

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def chunk_id(text: str, source: str) -> str:
    return hashlib.sha1((text + source).encode()).hexdigest()
