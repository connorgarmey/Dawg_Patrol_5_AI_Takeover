import time
import psutil
import pandas as pd
from search2 import get_embedding, search_chroma, search_redis, generate_rag_response, search_embeddings, search_faiss
from ingest2 import clear_redis_store, create_hnsw_index, process_pdfs, clear_chroma_store, clear_faiss_store
import os
import pickle 

ally = "/Users/alisonpicerno/Desktop/ds 4300/Dawg_Patrol_5_AI_Takeover/data"
connor = "/Users/connorgarmey/Documents/Large Scale/Practical 2/Dawg_Patrol_5_AI_Takeover/data"
data_path = ally

# Define different chunk sizes and models
top_k_values = [2,5,10] 
models = ["chroma", "redis", "faiss"]
query1 = "Explain the benefit of using B+ Trees in 2 sentences"
query2 = "What is the final state of an AVL tree with the numbers 27, 23, and 21"
query3 = "How was the battle of Gettysburg won?"
query4 = "Me fi do that and why should it be said?"
query = "What are the four components of ACID compliance?"

# Store results in a list
benchmark_results = []
chunk_sizes = [200, 500]
overlap = [0, 10, 50]
 
# Define which models to test 
embedding_models = ["snowflake-arctic-embed"]

# Function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024**2  # Convert to MB

# Benchmark function
def benchmark_search(query, model="faiss", k_size=3):
    results = {}

    # Track memory before
    memory_before = get_memory_usage()
    start_time = time.time()

    # Get embedding
    query_embedding = get_embedding(query)
    embedding_time = time.time() - start_time

    search_start = time.time()
    if model == "redis":
        print('search started for redis')
        retrieved_results = search_redis(query_embedding, top_k=k_size)
        print('search for redis ended')
    elif model == "chroma":
        print('search started for chroma')
        retrieved_results = search_chroma(query_embedding, top_k=k_size)
        print('search for chroma ended')
    elif model == "faiss":
        print('search started for faiss')
        retrieved_results = search_faiss(query_embedding, top_k=k_size)
        print('search for faiss ended')

    else:
        return "Invalid model"

    search_time = time.time() - search_start
    memory_after = get_memory_usage()

    # Generate response using retrieved results
    response_start = time.time()
    print('getting response')
    model_response = generate_rag_response(query, retrieved_results)
    print('model response retrieved')
    response_time = time.time() - response_start

    # Store results in dictionary
    results["embedding_model"] = embed_model
    results["chunk_size"] = chunk_size
    results['overlap'] = ov
    results["model"] = model
    results["top_k_size"] = k_size
    results["embedding_time"] = round(embedding_time, 4)
    results["search_time"] = round(search_time, 4)
    results["response_time"] = round(response_time, 4)
    results["memory_usage"] = round(memory_after - memory_before, 2)
    results["model_response"] = model_response  # Store model's generated respons

    return results

for chunk_size in chunk_sizes:
    for ov in overlap:
        for embed_model in embedding_models:
            # Re-run ingestion for each chunk size
            
            print(f"\n--- Ingesting PDFs with chunk_size={chunk_size}, overlap={overlap} ---")
            if "redis" in models:
                clear_redis_store()  # Clear Redis only if using Redis
                create_hnsw_index()

            if "faiss" in models:
                clear_faiss_store()  # Reset FAISS only if using FAISS

            if "chroma" in models:
                clear_chroma_store()  # Reset Chroma only if using Chroma
                
            create_hnsw_index()
            process_pdfs(data_path, chunk_size=chunk_size, overlap=ov, models=models, embedding_model=embed_model)  
            

            for model in models:
                for k in top_k_values:
                    result = benchmark_search(query, model=model, k_size=k)
                    #answer = generate_rag_response(query, search_embeddings(query), vector_model = model)
                    benchmark_results.append(result)
                    #benchmark_results.append(answer)

# Convert to DataFrame for easy analysis
df_results = pd.DataFrame(benchmark_results)
print(df_results)

# Save results to CSV for later use
df_results.to_csv("benchmark_results.csv", index=False)