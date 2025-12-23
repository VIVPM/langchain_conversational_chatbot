import pdfplumber
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def table_to_markdown(table_data):
    """Converts a list of lists (table data) into a Markdown table string."""
    if not table_data:
        return ""
    
    # Filter out empty rows or rows with all None
    table_data = [row for row in table_data if any(cell for cell in row)]
    if not table_data:
        return ""

    # Ensure all rows have the same number of columns
    num_cols = max(len(row) for row in table_data)
    table_data = [list(row) + [""] * (num_cols - len(row)) for row in table_data]
    
    # Sanitize cell content (remove newlines to keep markdown table structure valid)
    sanitized_data = []
    for row in table_data:
        sanitized_row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        sanitized_data.append(sanitized_row)

    if not sanitized_data:
        return ""

    # Create header and separator
    header = sanitized_data[0]
    separator = ["---"] * num_cols
    
    # Build the table
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    
    for row in sanitized_data[1:]:
        lines.append("| " + " | ".join(row) + " |")
        
    return "\n".join(lines)

def summarize_table(llm, markdown_table):
    """Uses the LLM to summarize the table content."""
    if not llm:
        return "Table found (LLM not available for summary)."
        
    prompt = PromptTemplate.from_template(
        """You are an expert data analyst. Summarize the following table.
        Highlight the key trends, important data points, and the overall purpose of the table.
        Keep the summary concise but informative.

        Table:
        {table}

        Summary:"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="summary")
    try:
        res = chain({"table": markdown_table})
        return res["summary"].strip()
    except Exception as e:
        return f"Error summarizing table: {e}"

def extract_pdf_content(file_stream, llm=None) -> str:
    """
    Extracts text and tables from a PDF stream.
    Tables are converted to Markdown and summarized by the LLM.
    """
    full_text = []
    
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # 1. Extract Tables
                tables = page.extract_tables()
                page_content = []
                
                if tables:
                    for table in tables:
                        md_table = table_to_markdown(table)
                        if md_table:
                            summary = summarize_table(llm, md_table)
                            # Format: Summary + Raw Data
                            table_block = (
                                f"\n\n**Table Summary (Page {page_num + 1}):**\n{summary}\n\n"
                                f"**Table Data:**\n{md_table}\n\n"
                            )
                            page_content.append(table_block)
                
                # 2. Extract Text
                # We extract text and try to filter out the table text to avoid duplication? 
                # Actually, pdfplumber's extract_text() usually includes table text. 
                # Ideally we'd filter it, but for now, having it twice (once in raw text, once in structured table) 
                # is acceptable and might even help retrieval. 
                # However, to be cleaner, we can just append the raw text.
                # Using x_tolerance=1 to prevent words from being joined together (default is 3)
                text = page.extract_text(x_tolerance=1)
                if text:
                    page_content.append(text)
                
                full_text.append("\n".join(page_content))
                
    except Exception as e:
        return f"Error processing PDF: {e}"

    return "\n\n".join(full_text)
