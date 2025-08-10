
import streamlit as st
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import SambaNovaCloud
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import json
from langchain_community.utilities import GoogleSerperAPIWrapper
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import hashlib

load_dotenv(dotenv_path='../crewai/.env')

# Get Supabase credentials from environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Initialize Supabase client if credentials are available
supabase: Client = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    st.error("Supabase credentials not found in environment variables.")

# Main app
st.title("Research Assistant App")

def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize session states
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "show_login" not in st.session_state:
    st.session_state.show_login = True

if not st.session_state.logged_in:
    if st.session_state.show_signup:
        st.subheader("Signup")
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Signup")
            if submit:
                if username and password:
                    existing_user = supabase.table("profiles").select("*").eq("username", username).execute()
                    if existing_user.data:
                        st.error("Username already exists.")
                    else:
                        password_hash = hash_password(password)
                        supabase.table("profiles").insert({
                            "username": username,
                            "password": password_hash,
                            "chat_history": []
                        }).execute()
                        st.success("Signup successful! Please login.")
                        st.session_state.show_signup = False
                        st.session_state.show_login = True
                        st.rerun()
                else:
                    st.error("Please fill in all fields.")

    if st.session_state.show_login:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if username and password:
                    user = supabase.table("profiles").select("*").eq("username", username).execute()
                    if user.data and hash_password(password) == user.data[0]["password"]:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user.data[0]["id"]
                        st.session_state.username = username
                        st.success("Login successful!")
                        st.session_state.show_login = False
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Please fill in all fields.")
        if st.button("Create new account"):
            st.session_state.show_login = False
            st.session_state.show_signup = True
            st.rerun()

else:
    # Sidebar for API keys
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("SambaNova API Key", type="password")
    serper_api_key = st.sidebar.text_input("Serper API Key", type="password")

    st.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.messages = []
        st.session_state.memory = None
        st.session_state.show_login = True
        st.session_state.show_signup = False
        st.rerun()

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    # Button to process documents
    if st.button("Process Documents") and uploaded_files and api_key:
        with st.spinner("Processing documents..."):
            all_docs = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            for uploaded_file in uploaded_files:
                text = extract_text_from_pdf(uploaded_file)
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    all_docs.append(Document(page_content=chunk, metadata={"source": uploaded_file.name}))
            
            # Embeddings using HuggingFace
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            
            # Create FAISS index
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.success("Documents processed and indexed successfully!")

    # Load chat history from Supabase
    if "messages" not in st.session_state:
        try:
            response = supabase.table("profiles").select("chat_history").eq("id", st.session_state.user_id).single().execute()
            st.session_state.messages = response.data["chat_history"] or []
        except Exception as e:
            st.error(f"Error loading chat history: {str(e)}")
            st.session_state.messages = []

    # Initialize memory if not present, and populate from loaded messages
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        # Populate memory with historical messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.session_state.memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "assistant":
                st.session_state.memory.chat_memory.add_ai_message(message["content"])

    # Sidebar for chat history
    st.sidebar.title("Chat History")
    if st.sidebar.button("Load History"):
        st.session_state.show_history = True

    if "show_history" in st.session_state and st.session_state.show_history:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Query section as chatbot
    if "vectorstore" in st.session_state and api_key and serper_api_key:
        query = st.chat_input("Your question")
        if query:
            # Append user message to history
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.spinner("Generating answer..."):
                # Initialize SambaNova LLM
                llm = SambaNovaCloud(
                    model="Meta-Llama-3.3-70B-Instruct",
                    sambanova_api_key=api_key
                )
                
                # First, try to answer from memory (conversation history) without retrieval
                memory_prompt = PromptTemplate.from_template(
                    "Based only on the following conversation history, answer the question if possible. If the information is not in the history, say 'I need to retrieve more information.'\n\nChat History:\n{chat_history}\n\nQuestion: {question}\n\nAnswer:"
                )
                memory_chain = LLMChain(
                    llm=llm,
                    prompt=memory_prompt,
                    memory=st.session_state.memory,
                    output_key="answer"
                )
                memory_response = memory_chain({"question": query})
                memory_answer = memory_response['answer']
                
                if "I need to retrieve more information" in memory_answer:
                    # Fall back to retrieval if not found in memory
                    condense_prompt = PromptTemplate.from_template(
                        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
                    )
                    # Custom prompt for the combine_docs_chain to detect no info
                    combine_prompt = PromptTemplate.from_template(
                        "Use the following pieces of context to answer the question at the end. "
                        "If you cannot answer the question based on the context, just say 'Information not found in the provided documents.' "
                        "Do not try to make up an answer.\n\n{context}\n\nQuestion: {question}\n\nHelpful Answer:"
                    )
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10}, search_type="mmr"),
                        memory=st.session_state.memory,
                        return_source_documents=True,
                        condense_question_prompt=condense_prompt,
                        combine_docs_chain_kwargs={"prompt": combine_prompt},
                        output_key="answer"
                    )
                    
                    # Get response from retrieval chain
                    response = qa_chain({"question": query})
                    answer = response['answer']
                    source_documents = response['source_documents']
                    
                    # Check if retrieval couldn't answer
                    if "Information not found in the provided documents." in answer:
                        if not serper_api_key:
                            answer = "Serper API key is required for web search fallback."
                            sources = []
                        else:
                            # Fallback to web search using Serper
                            search_tool = GoogleSerperAPIWrapper(serper_api_key=serper_api_key, k=5)  # Limit to 5 results for efficiency
                            search_results_dict = search_tool.results(query)  # Get dict of results
                            
                            # Compile snippets as search_results string
                            organic_results = search_results_dict.get('organic', [])
                            search_results = "\n".join([f"Snippet: {res['snippet']}\nLink: {res['link']}" for res in organic_results])
                            
                            # Prompt for summarizing/answering based on search results
                            web_prompt = PromptTemplate.from_template(
                                "Based on the following web search results, provide a concise answer to the question. "
                                "Include key sources.\n\nSearch Results:\n{search_results}\n\nQuestion: {question}\n\nAnswer:"
                            )
                            web_chain = LLMChain(llm=llm, prompt=web_prompt)
                            web_response = web_chain({"question": query, "search_results": search_results})
                            answer = web_response['text']
                            
                            # Extract sources (links) from organic results
                            sources = [res['link'] for res in organic_results]
                            
                        # Format assistant response for web fallback
                        assistant_response = f"**Answer (from web search):**\n{answer}\n\n**Web Sources used:**\n" + "\n".join([f"- {source}" for source in sources])
                    else:
                        # Format assistant response with document sources
                        sources = set(doc.metadata['source'] for doc in source_documents)
                        assistant_response = f"**Answer:**\n{answer}\n\n**Sources used:**\n" + "\n".join([f"- {source}" for source in sources])
                else:
                    # Use answer from memory
                    assistant_response = f"**Answer (from memory):**\n{memory_answer}"
                    sources = []  # No sources since from memory
                
                # Append assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
                
                # Save chat history to Supabase
                try:
                    supabase.table("profiles").update({
                        "chat_history": st.session_state.messages
                    }).eq("id", st.session_state.user_id).execute()
                except Exception as e:
                    st.error(f"Error saving chat history: {str(e)}")
                
    else:
        if not api_key or not serper_api_key:
            st.info("Please enter your SambaNova API key and Serper API key in the sidebar.")
        elif not uploaded_files:
            st.info("Please upload PDF files and process them.")
        else:
            st.info("Click 'Process Documents' to index the uploaded files.")
