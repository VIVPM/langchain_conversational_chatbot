# QuickBrain

An AI-powered document Q&A chatbot with RAG (Retrieval-Augmented Generation), web search, and persistent chat history.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Login/    │  │   File      │  │   Chat      │  │   Settings          │ │
│  │   Signup    │  │   Upload    │  │   Window    │  │   (Model, API Keys) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROCESSING LAYER                                 │
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │  Input Guardrails │    │   Hybrid Memory  │    │   Query Router       │  │
│  │  (Detoxify)       │    │  (Summary +      │    │                      │  │
│  │                   │    │   Recent Msgs)   │    │  ┌────────────────┐  │  │
│  └──────────────────┘    └──────────────────┘    │  │ Has Documents? │  │  │
│                                                   │  └───────┬────────┘  │  │
│                                                   │      Yes │ No        │  │
│                                                   │          ▼           │  │
│                                                   │  ┌──────────────┐    │  │
│                                                   │  │ Web Search?  │    │  │
│                                                   │  └──────────────┘    │  │
│                                                   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐
│     RAG PIPELINE     │  │   WEB SEARCH     │  │    DIRECT ANSWER         │
│                      │  │                  │  │                          │
│  ┌────────────────┐  │  │  Google Serper   │  │  LLM with conversation   │
│  │ PDF Processor  │  │  │  API (5 results) │  │  context from memory     │
│  │ (pdfplumber +  │  │  │                  │  │                          │
│  │  table summary)│  │  └──────────────────┘  └──────────────────────────┘
│  └────────────────┘  │
│         ▼            │
│  ┌────────────────┐  │
│  │ Text Splitter  │  │
│  │ (512 chunks)   │  │
│  └────────────────┘  │
│         ▼            │
│  ┌────────────────┐  │
│  │ BGE Embeddings │  │
│  └────────────────┘  │
│         ▼            │
│  ┌────────────────┐  │
│  │ Pinecone       │  │
│  │ Vector Store   │  │
│  │ (MMR, k=10)    │  │
│  └────────────────┘  │
└──────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM LAYER (SambaNova)                             │
│                                                                             │
│   Llama 3.3 │ Llama 4 │ DeepSeek V3/R1 │ Qwen3 │ ALLaM │ E5-Mistral       │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER                                  │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │         Supabase            │    │            Pinecone                 │ │
│  │  • User profiles            │    │  • Document embeddings              │ │
│  │  • Chat history (JSON)      │    │  • Per-user namespaces              │ │
│  │  • Authentication           │    │                                     │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-format Document Ingestion**: PDF (with table extraction), TXT, DOCX
- **Hybrid RAG**: Vector search via Pinecone with MMR for diversity
- **Web Search**: Optional Google Serper integration for real-time information
- **Hybrid Memory**: Keeps recent messages verbatim, summarizes older context
- **Content Moderation**: Detoxify-based input guardrails
- **Multi-model Support**: 12+ LLMs via SambaNova Cloud
- **Persistent Chats**: Supabase-backed conversation history

## Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd quickbrain
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
```

### 3. Run

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Main Streamlit application
├── config.py                 # Environment variables and constants
├── services/
│   ├── ai_processor.py       # LLM, embeddings, RAG, guardrails
│   ├── authentication.py     # Supabase auth (signup/login)
│   ├── chat_manager.py       # Session state, hybrid memory
│   └── pdf_processor.py      # PDF extraction with table handling
├── ui/
│   └── components.py         # Sidebar, chat list, settings UI
├── utils/
│   └── common_utils.py       # Hashing, file utils, rate limiting
└── artifacts/                # (if applicable) stored models
```

## Configuration

| Variable | Description |
|----------|-------------|
| `MAX_FILES` | Max uploadable files per session (default: 3) |
| `MAX_FILE_SIZE_BYTES` | File size limit (default: 2MB) |
| `DEFAULT_MODEL` | Fallback LLM model |
| `TIMEZONE` | Timestamp timezone (default: Asia/Kolkata) |

## API Keys Required

| Service | Purpose | Get it from |
|---------|---------|-------------|
| SambaNova | LLM inference | [SambaNova Cloud](https://cloud.sambanova.ai) |
| Pinecone | Vector storage | [Pinecone](https://pinecone.io) |
| Supabase | Auth + chat persistence | [Supabase](https://supabase.com) |
| Serper (optional) | Web search | [Serper](https://serper.dev) |

## How It Works

1. **Upload Documents** → PDF/TXT/DOCX files are chunked (512 tokens, 128 overlap)
2. **Embed & Index** → BGE-large embeddings stored in Pinecone under user namespace
3. **Query** → User question checked for toxicity, then routed:
   - If documents indexed → RAG retrieval (MMR, k=10)
   - If web search enabled → Serper API fetch
   - Else → Direct LLM response with conversation context
4. **Memory** → Recent 6 messages kept verbatim; older messages summarized after 15 total

## License

MIT