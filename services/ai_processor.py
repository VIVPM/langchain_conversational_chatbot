import re
from typing import Tuple, Dict
from detoxify import Detoxify
from langchain_community.llms import SambaNovaCloud
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.docstore.document import Document
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from utils.common_utils import chunk_id, extract_text_from_file

# --- LLM ---
def get_llm(model_name: str, api_key: str):
    return SambaNovaCloud(model=model_name, sambanova_api_key=api_key)

def answer_direct(llm, history_text: str, question: str) -> str:
    prompt = PromptTemplate.from_template(
        """Answer the user's question.
        If Conversation Facts are provided, use them only to resolve references or small gaps. Do not invent details.

        Conversation Facts:
        {facts_context}

        Question: {q}

        Answer:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="ans")
    return chain({"q": question, "facts_context": history_text})["ans"].strip()

def answer_from_context(llm, context: str, question: str) -> str:
    prompt = PromptTemplate.from_template(
        "Use the context to answer. If insufficient, say 'Information not found in the provided documents.'\n\n{context}\n\nQuestion: {q}\n\nAnswer:"
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="ans")
    return chain({"context": context, "q": question})["ans"].strip()

# --- Vectorstore ---
def make_vectorstore(username: str):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
        embedding=embeddings,
        namespace=username,
    )

from services.pdf_processor import extract_pdf_content

# --- Ingest ---
def split_files(uploaded_files, llm=None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    all_docs = []
    for uf in uploaded_files or []:
        name = uf.name.lower()
        if name.endswith(".pdf"):
            text = extract_pdf_content(uf, llm)
        else:
            text = extract_text_from_file(uf)
            
        for chunk in splitter.split_text(text):
            all_docs.append(Document(page_content=chunk, metadata={"source": uf.name}))
    return all_docs

def index_docs(vectorstore, docs):
    if not docs:
        return 0
    ids = [chunk_id(d.page_content, d.metadata.get("source", "")) for d in docs]
    vectorstore.add_documents(docs, ids=ids)
    return len(docs)

from firecrawl import FirecrawlApp

# --- Search ---
def web_search_answer(llm, firecrawl_api_key: str, question: str) -> tuple[str, list[str]]:
    app = FirecrawlApp(api_key=firecrawl_api_key)
    try:
        # Using 'search' mode to get results similar to Google Search
        sr = app.search(question,limit=5)
        
        # Firecrawl returns an object with a 'data' or 'web' attribute containing the list of results.
        # Based on testing, it returns an object with a 'web' attribute.
        if hasattr(sr, 'web'):
            results_data = sr.web
        elif hasattr(sr, 'data'):
            results_data = sr.data
        elif isinstance(sr, dict):
            results_data = sr.get('data', sr.get('web', []))
        else:
            results_data = []
        
        search_results = "\n".join(
            f"Title: {getattr(r, 'title', 'No Title')}\nLink: {getattr(r, 'url', getattr(r, 'link', ''))}\nSnippet: {getattr(r, 'description', getattr(r, 'snippet', ''))}\n"
            for r in results_data
        )
        
        prompt = PromptTemplate.from_template(
            "Based on these web search results, answer concisely and include key sources.\n\n{results}\n\nQuestion: {q}\n\nAnswer:"
        )
        chain = LLMChain(llm=llm, prompt=prompt, output_key="ans")
        ans = chain({"results": search_results, "q": question})["ans"].strip()
        
        sources = [getattr(r, 'url', getattr(r, 'link', '')) for r in results_data if getattr(r, 'url', None) or getattr(r, 'link', None)]
        return ans, sources
    except Exception as e:
        return f"Error performing web search: {str(e)}", []

# --- Guardrails ---
class InputGuardrails:
    """Guardrails for validating user input before LLM processing"""

    # Lazy-loaded toxicity model
    _toxicity_model = None

    @classmethod
    def get_toxicity_model(cls):
        if cls._toxicity_model is None:
            cls._toxicity_model = Detoxify("original")
        return cls._toxicity_model

    @staticmethod
    def check_toxicity_and_explicit(query: str) -> Tuple[bool, str]:
        cutoff = 0.05
        try:
            model = InputGuardrails.get_toxicity_model()
            scores: Dict[str, float] = model.predict(query) or {}
        except Exception:
            scores = {}

        tripped = [k for k, v in scores.items() if v > cutoff]

        if tripped:
            human_labels = [k.replace("_", " ") for k in tripped]
            return False, f"Your question contains inappropriate content ({', '.join(human_labels)}). Please rephrase respectfully."

        return True, ""
