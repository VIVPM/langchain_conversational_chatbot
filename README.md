# QuickBrain

Streamlit RAG chat app with Pinecone, SambaNova LLM, and Supabase-backed auth.

## Summary

- Upload PDF, DOCX, TXT files. Text is extracted, chunked, embedded with `BAAI/bge-large-en-v1.5`, and indexed in **Pinecone**.
- Ask questions in chat. If web search is enabled, the app uses web results; else it tries RAG over Pinecone (if your docs are indexed); default it answers directly with the LLM using conversation facts.
- Users sign up and log in. Passwords are hashed and profiles plus chats are stored in **Supabase**.
- Sidebar controls: API keys, web-search toggle, document processing, chat search, and new chat.

## Architecture

```mermaid
flowchart LR
  U[User: Streamlit UI] -->|Login/Signup| SB[(Supabase profiles)]
  U -->|Upload PDFs| PDF[PyMuPDF]
  PDF --> SPLIT[TextSplitter 512/128]
  SPLIT --> EMB[HF Embeddings bge-large]
  EMB --> IDX[(Pinecone Index\nnamespace = username)]
  U -->|Ask| Q[Question]
  Q --> WEB{Web search enabled?}
  WEB -- Yes --> SERP[Serper k=5] --> WSUM[LLM summarize]
  WEB -- No --> RET{Docs indexed?}
  RET -- Yes --> R[VDB Retriever k=10, MMR] --> COMB[LLM combine]
  RET -- No --> DIRECT[Direct LLM\n(with conversation facts)]
  WSUM --> OUT[Answer+Sources]
  COMB --> OUT
  DIRECT --> OUT
  OUT --> SB
  OUT --> U
```

## Key Components

- **Streamlit UI**: chat interface, sidebar settings, chat list and search.
- **Auth + Persistence**: Supabase table `profiles` keeps `username`, `password` (SHA‑256 hash), and a list of `chats`. Chats store `id`, `title`, `messages`, `created_at`, `updated_at`.
- **File ingestion**: PDF via `PyMuPDF`; TXT via UTF-8 decode; DOCX via `python-docx`. Uses `RecursiveCharacterTextSplitter` with `chunk_size=512`, `chunk_overlap=128`.
- **Embeddings**: `HuggingFaceEmbeddings("BAAI/bge-large-en-v1.5")`.
- **Vector store**: **Pinecone** via `PINECONE_INDEX_NAME` and `PINECONE_API_KEY`. Namespace per user = `username`. IDs are SHA‑1 of content + source.
- **LLM**: `SambaNovaCloud` with selectable models in the sidebar (default: `DeepSeek-V3.1`).
- **Answering flow**:
  1) If web search is toggled and Serper key is provided, use web results with LLM summarization.  
  2) Else, if your docs are indexed, retrieve with Pinecone (k=10, MMR) and answer from context.  
  3) Else, answer directly with the LLM, using conversation facts from memory.
- **Guardrails**: Detoxify-based input screening flags toxic content and adds a warning to the assistant response.

## Requirements

- Python 3.10+
- Accounts/keys:
  - **Supabase**: `SUPABASE_URL`, `SUPABASE_KEY`
  - **Pinecone**: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
  - **SambaNova** API key (entered in sidebar for answering). For conversation summarization memory, set environment variable `SAMBANOVA_API_KEY`.
  - **Serper** API key for optional web search (entered in sidebar)
  - Username constraint: must be a Gmail address (e.g., `name@gmail.com`).

## Installation

```bash
pip install -r requirements.txt
```

> If you use GPU embeddings or other models, install the matching extras separately.

## Configuration

Create a `.env` that the app loads. By default the code calls in `config.py`:
```python
load_dotenv(dotenv_path=".env")
```
Options:
- Put your variables in `.env`, or
- Change the `dotenv_path` in `config.py` to your desired location.

Minimum variables in .env file:
```
SUPABASE_URL=...
SUPABASE_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
SAMBANOVA_API_KEY=...
```

## Running

```bash
streamlit run app.py
```

In the sidebar:
1. Paste SambaNova API key. Optionally paste Serper API key.
2. Toggle **Web search** if you want web fallback.
3. Upload PDFs and click **Process Documents**.
4. Start chatting. Use **Chats** to create or open a conversation. Use **Search chats** to filter by message text.

## Notes and Defaults

- Chunking: 512 characters, 128 overlap.
- Retrieval: MMR with `k=10`.
- Pinecone namespace: the current `username`.
- Password hashing: SHA‑256 for demo. For production, prefer a dedicated auth provider or salted password hashing (e.g., Argon2/bcrypt) with rate‑limiting and MFA.
- If Serper key is missing or web search is off, the app falls back to RAG or direct LLM.
- Timezone: `Asia/Kolkata` by default. Change `TIMEZONE` in `config.py` if needed.

## Troubleshooting

- **Pinecone connection**: ensure `PINECONE_INDEX_NAME` and `PINECONE_API_KEY` are set. The app uses the index name (not host).
- **“Supabase credentials not found”**: set `SUPABASE_URL` and `SUPABASE_KEY` in the `.env` that `config.py` loads.
- **No answer from docs**: confirm documents were processed, embeddings created, and the correct namespace is used.
- **Serper missing**: web search button can be ON, but without a Serper key the code will skip web and use LLM only.
- **SambaNova summarization**: conversation summary memory uses `SAMBANOVA_API_KEY` from environment. Set it if summaries are not updating.

## License

MIT. See `LICENSE`.