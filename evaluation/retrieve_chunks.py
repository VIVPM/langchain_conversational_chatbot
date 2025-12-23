import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from pinecone import Pinecone

def retrieve_chunks():
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}' with namespace 'viv@gmail.com'...")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
        embedding=embeddings,
        namespace="viv@gmail.com"
    )
    
    # Direct query with dummy vector to fetch chunks
    print("Querying Pinecone directly...")
    try:

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Get dimension from stats or assume 1024 (bge-large-en-v1.5)
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        # We need to know the dimension. Usually 1024 for this model.
        # If we can't get it easily, we'll try 1024.
        dimension = 1024 
        if 'dimension' in stats:
             dimension = stats['dimension']
        elif 'index_dimension' in stats: # older api
             dimension = stats['index_dimension']
             
        dummy_vector = [0.0] * dimension
        
        # Query for 1000 vectors (max limit usually)
        results = index.query(
            vector=dummy_vector,
            top_k=1000,
            namespace="viv@gmail.com",
            include_metadata=True
        )
        
        output_data = []
        for match in results['matches']:
            if match.metadata:
                output_data.append({
                    "id": match.id,
                    "content": match.metadata.get('text', '') or match.metadata.get('page_content', ''),
                    "metadata": match.metadata
                })
        
        output_file = "retrieved_chunks.json"
        print(f"Saving {len(output_data)} chunks to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print("Done!")
        
    except Exception as e:
        print(f"Error during direct query: {e}")

if __name__ == "__main__":
    retrieve_chunks()
