# LangChain Conversational Chatbot

## Research Assistant App

### Overview

This is a Streamlit-based chatbot application that acts as a research assistant. It lets you upload multiple PDF documents, extracts and indexes their content, and allows you to have a conversational chat about them. The app uses **LangChain** for Retrieval-Augmented Generation (RAG), **SambaNova's Meta-Llama-3.3-70B-Instruct LLM** for generating responses, and a conversational memory to maintain context. If the answer isn't found in the documents, it falls back to a web search using **Serper**. Your chat history is saved to a JSON file, so you can continue conversations across sessions.

### Key Features

  * **PDF text extraction** using `PyMuPDF`.
  * **Document chunking and embedding** with HuggingFace models (`BAAI/bge-large-en-v1.5`).
  * **Vector database** with **FAISS** for efficient similarity search, using Maximum Marginal Relevance (MMR) for diverse results.
  * **Conversational chain with memory** to handle follow-up questions effectively.
  * **Fallback mechanism:** The app first checks its memory, then retrieves from the uploaded documents, and finally searches the web if needed.
  * **Persistent chat history** is saved to `chat_history.json`.

### Requirements

  * Python 3.8+
  * **API Keys:**
      * SambaNova API key (for the LLM).
      * Serper API key (for web search fallback).

### Required Libraries

Install the dependencies using the following command:

```
pip install streamlit pymupdf langchain-huggingface langchain-community faiss-cpu langchain
```

> **Note:** This setup uses the CPU-only version of FAISS. For GPU acceleration, install `faiss-gpu` instead.

### Installation

1.  Clone or download the repository containing the app code.
2.  Install the required libraries as listed above.
3.  Make sure you have valid API keys for SambaNova and Serper.

### Usage

1.  Run the app from your terminal:
    ```
    streamlit run app.py
    ```
    (Replace `app.py` with your file name.)
2.  In the app's sidebar, enter your **SambaNova API key** and your **Serper API key**. The Serper key is necessary for web search fallback.
3.  Upload your PDF files using the file uploader.
4.  Click **"Process Documents"** to extract text, chunk, embed, and index your PDFs.
5.  Once the documents are processed, use the chat input at the bottom to ask questions.

The app will follow this logic:

  * First, it will try to answer from your current conversation history (memory).
  * If not found, it will retrieve relevant information from your uploaded documents.
  * If the information is still not available, it will perform a web search using Serper and provide a summarized answer.

Your chat history is automatically displayed and saved to `chat_history.json` for persistence.

### Example Workflow

1.  Upload several research papers on a specific topic, like AI algorithms.
2.  Ask a question: "What is the Artificial Bee Colony algorithm?"
3.  Ask a follow-up question: "How is it applied in IoT?"
4.  If the answer isn't in your documents, the app will search the web and provide a sourced answer.

### Code Structure

  * **PDF Processing:** Extracts text with `PyMuPDF`, chunks text with `RecursiveCharacterTextSplitter`, embeds with a HuggingFace model, and indexes the content in FAISS.
  * **Memory Handling:** Uses `ConversationBufferMemory` to store context and loads/saves history from a JSON file.
  * **Query Flow:**
      * Checks memory first using `LLMChain`.
      * Retrieves from documents using `ConversationalRetrievalChain`.
      * Performs web fallback with `GoogleSerperAPIWrapper`.
  * **UI:** Built with a Streamlit chat interface, including spinners for loading states.

### Contributing

Feel free to fork the repository and submit pull requests. You can contribute improvements such as adding more embedding models or advanced reranking techniques.

### License

This project is under the **MIT License**. See `LICENSE` for more details.