from langchain_community.llms import SambaNovaCloud
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_llm(model_name: str, api_key: str):
    return SambaNovaCloud(model=model_name, sambanova_api_key=api_key, temperature=0.4, top_p=0.8, top_k=1000)

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
