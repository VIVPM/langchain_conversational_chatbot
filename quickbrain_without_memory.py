# without memory

import streamlit as st
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import SambaNovaCloud
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Sidebar for API key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("SambaNova API Key", type="password")

# Main app
st.title("Research Assistant App")

# Upload multiple PDFs
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
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create FAISS index
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        
        # Store in session state
        st.session_state.vectorstore = vectorstore
        st.success("Documents processed and indexed successfully!")

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query section as chatbot
if "vectorstore" in st.session_state and api_key:
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
            
            # Prepare chat history for the chain (list of tuples: (user_msg, assistant_msg))
            chat_history = []
            for i in range(0, len(st.session_state.messages) - 1, 2):  # Pair user and assistant
                if i + 1 < len(st.session_state.messages):
                    chat_history.append((st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"]))
            
            # ConversationalRetrievalChain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10}, search_type="mmr"),
                return_source_documents=True,
                output_key="answer"
            )
            
            # Get response, passing chat_history
            response = qa_chain({"question": query, "chat_history": chat_history})
            print(response)
            answer = response['answer']  # Note: Key is 'answer' in ConversationalRetrievalChain
            source_documents = response['source_documents']
            
            # Extract unique sources
            sources = set(doc.metadata['source'] for doc in source_documents)
            
            # Format assistant response
            assistant_response = f"**Answer:**\n{answer}\n\n**Sources used:**\n" + "\n".join([f"- {source}" for source in sources])
            
            # Append assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
else:
    if not api_key:
        st.info("Please enter your SambaNova API key in the sidebar.")
    elif not uploaded_files:
        st.info("Please upload PDF files and process them.")
    else:
        st.info("Click 'Process Documents' to index the uploaded files.")