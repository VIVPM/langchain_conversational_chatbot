import fitz

def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        text = ""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type")
