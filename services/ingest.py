from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils.crypto import chunk_id
from utils.files import extract_text_from_file

def split_files(uploaded_files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    all_docs = []
    for uf in uploaded_files or []:
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
