import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# App limits
MAX_FILES = 3
MAX_FILE_SIZE_BYTES = 2000 * 1024
DEFAULT_CHAT_PAGE_SIZE = 10

# UI/LLM defaults
DEFAULT_MODEL = "DeepSeek-V3.1"
TIMEZONE = "Asia/Kolkata"
