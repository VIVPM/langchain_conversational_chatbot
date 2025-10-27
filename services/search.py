from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def web_search_answer(llm, serper_api_key: str, question: str) -> tuple[str, list[str]]:
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
