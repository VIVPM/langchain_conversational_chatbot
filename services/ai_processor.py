import re
import streamlit as st
from typing import Tuple, Dict
from detoxify import Detoxify
from langchain_community.llms import SambaNovaCloud
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from utils.common_utils import chunk_id, extract_text_from_file
from services.pdf_processor import extract_pdf_content

# --- LLM ---
def get_llm(model_name: str, api_key: str):
    return SambaNovaCloud(model=model_name, sambanova_api_key=api_key)

def answer_direct(llm, history_text: str, question: str) -> str:
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant. Answer the user's question.

            IMPORTANT RULES:
            - Answer the question as if it were asked in isolation
            - If the question contains pronouns or references (like "it", "that", "this", "them") that need context, silently use the conversation history below to understand what they refer to
            - NEVER mention or reference the conversation history in your response
            - NEVER say things like "Based on the conversation" or "Since there's no information in the facts"
            - Just provide a direct, helpful answer to the question

            Conversation History (use ONLY to resolve references, do not mention):
            {facts_context}

            Question: {q}
        """
    )

    try:
        chain = LLMChain(llm=llm, prompt=prompt, output_key="ans")
        return chain({"q": question, "facts_context": history_text})["ans"].strip()
    except Exception as e:
        return f"Sorry, I encountered an error while generating a response. Please try again. (Error: {str(e)[:100]})"

def answer_from_context(llm, context: str, question: str) -> str:
    prompt = PromptTemplate.from_template(
        "Use the context to answer. If insufficient, say 'Information not found in the provided documents.'\n\n{context}\n\nQuestion: {q}\n\nAnswer:"
    )
    try:
        chain = LLMChain(llm=llm, prompt=prompt, output_key="ans")
        return chain({"context": context, "q": question})["ans"].strip()
    except Exception as e:
        return f"Error retrieving answer from documents: {str(e)[:100]}"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# --- Vectorstore ---
def make_vectorstore(username: str):
    try:
        embeddings = get_embeddings()
        return PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY,
            embedding=embeddings,
            namespace=username,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Pinecone: {str(e)[:100]}")


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
    try:
        ids = [chunk_id(d.page_content, d.metadata.get("source", "")) for d in docs]
        vectorstore.add_documents(docs, ids=ids)
        return len(docs)
    except Exception as e:
        raise RuntimeError(f"Failed to index documents: {str(e)[:100]}")

# --- Search ---
def web_search_answer(llm, serper_api_key: str, question: str) -> tuple[str, list[str]]:
    try:
        search_tool = GoogleSerperAPIWrapper(serper_api_key=serper_api_key, k=5)
        sr = search_tool.results(question)
        organic = sr.get("organic", [])
        search_results = "\n".join(
            f"Snippet: {o.get('snippet','')}\nLink: {o.get('link','')}" for o in organic
        )
        prompt = PromptTemplate.from_template(
            "Based on these web search results, answer concisely and include key sources.\n\n{results}\n\nQuestion: {q}\n\nAnswer:"
        )
        chain = LLMChain(llm=llm, prompt=prompt, output_key="ans")
        ans = chain({"results": search_results, "q": question})["ans"].strip()
        sources = [o.get("link","") for o in organic if o.get("link")]
        return ans, sources
    except Exception as e:
        return f"Web search failed: {str(e)[:100]}", []

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
