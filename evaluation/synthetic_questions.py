import os
import json
import time
from dotenv import load_dotenv
from langchain_community.llms import SambaNovaCloud
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import load_dataset, Dataset
from langchain_community.llms import Ollama
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

# Load env vars
load_dotenv()

def load_chunk_content(file_path="retrieved_chunks.json"):
    """
    Reads the JSON file containing retrieved chunks and returns a list of their content.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        contents = [chunk.get("content", "") for chunk in chunks if "content" in chunk]
        
        print(f"Successfully loaded {len(contents)} chunks from {file_path}")
        return contents

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def evaluate_ragas():
    with open("synthetic_data_with_qa.json", "r", encoding="utf-8") as f:
        qagc_list = json.load(f)
            
    eval_dataset = Dataset.from_list(qagc_list[:1])
    local_llm = Ollama(model="phi3:mini",timeout=180)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={'normalize_embeddings': True})
    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    # print(eval_dataset)
    result = evaluate(
        eval_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config = RunConfig(
            max_workers=1,      # Single worker
            timeout=180         # 3 min timeout
        ),
        raise_exceptions=False
    )
    # print(result)
    results_df = result.to_pandas()
    results_df.to_csv("ragas_results.csv", index=False)
    print(results_df)

def generate_synthetic_data():
    # 1. Load Chunks
    all_contents = load_chunk_content()
    
    # Group chunks into sets of 3
    combined_chunks = []
    chunk_size = 3
    for i in range(0, len(all_contents), chunk_size):
        if i + chunk_size < len(all_contents):
            combined_chunks.append(all_contents[i:i + chunk_size])
        else:
            combined_chunks.append(all_contents[i:])
    
    selected_chunks = combined_chunks
    print(f"Created {len(selected_chunks)} combined chunks from {len(all_contents)} original chunks.")
    print(f"Processing all {len(selected_chunks)} combined chunks...")

    # 2. Initialize LLMs
    # Question Generator & Actual Answer Generator (8B)
    llm_8b = SambaNovaCloud(
        model="Meta-Llama-3.1-8B-Instruct",
        sambanova_api_key=os.getenv("SAMBANOVA_API_KEY"),
        max_tokens=1024,
        temperature=0.7
    )

    # Ground Truth Generator (70B)
    llm_70b = SambaNovaCloud(
        model="Meta-Llama-3.3-70B-Instruct",
        sambanova_api_key=os.getenv("SAMBANOVA_API_KEY"),
        max_tokens=1024,
        temperature=0.0 # Lower temp for ground truth
    )

    # 3. Define Prompts
    question_prompt_template = """\
You are a teacher preparing a test. Please create a question that can be answered by referencing the following context.

Context:
{context}

Question:"""
    question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["context"])

    ground_truth_prompt_template = """\
Use the following context and question to answer this question using *only* the provided context.

Question:
{question}

Context:
{context}

Answer:"""
    ground_truth_prompt = PromptTemplate(template=ground_truth_prompt_template, input_variables=["question", "context"])

    # 4. Define Chains (Legacy LLMChain style)
    # Chain to generate question from context
    question_chain = LLMChain(llm=llm_8b, prompt=question_prompt, output_key="question")
    
    # Chain to generate ground truth answer (70B)
    ground_truth_chain = LLMChain(llm=llm_70b, prompt=ground_truth_prompt, output_key="answer")

    # Chain to generate actual answer (8B)
    actual_answer_chain = LLMChain(llm=llm_8b, prompt=ground_truth_prompt, output_key="answer")

    # 5. Generation Loop
    synthetic_data = []
    
    output_file = "synthetic_data.json"
    
    for i, chunk_content in enumerate(selected_chunks):
        print(f"Processing chunk {i+1}/{len(selected_chunks)}...")
        
        try:
            # Generate Question
            # LLMChain returns a dict, we extract the output_key
            q_res = question_chain({"context": chunk_content})
            question = q_res["question"].strip()
            
            # Generate Ground Truth (70B)
            gt_res = ground_truth_chain({"question": question, "context": chunk_content})
            gt = gt_res["answer"].strip()
            
            # Generate Actual Answer (8B)
            ans_res = actual_answer_chain({"question": question, "context": chunk_content})
            ans = ans_res["answer"].strip()
            
            # Create data point
            data_point = {
                "contexts": chunk_content,
                "question": question,
                "ground_truth": gt,
                "answer": ans
            }
            synthetic_data.append(data_point)
            
            print(f"  Q: {question[:50]}...")
            
            # Sleep for 20 seconds
            time.sleep(20)
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Log error but continue? Or append error dict?
            # User didn't specify error handling for this format, but keeping it consistent
            error_point = {
                "contexts": chunk_content,
                "question": f"Error: {e}",
                "ground_truth": "Error",
                "answer": "Error"
            }
            synthetic_data.append(error_point)
            time.sleep(20)

    # 6. Save Results
    print(f"Saving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, indent=4, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    # generate_synthetic_data()
    evaluate_ragas()
