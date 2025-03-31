import time
import psutil
import pandas as pd
from search2 import get_embedding, search_chroma, search_redis, generate_rag_response, search_embeddings
from ingest2 import clear_redis_store, create_hnsw_index, process_pdfs

ally = "/Users/alisonpicerno/Desktop/ds 4300/Dawg_Patrol_5_AI_Takeover/data"
connor = "/Users/connorgarmey/Documents/Large Scale/Practical 2/Dawg_Patrol_5_AI_Takeover/data"
data_path = ally


# Function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024**2  # Convert to MB

# Benchmark function
def benchmark_search(query, model="chroma", chunk_size=3):
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
        retrieved_results = search_redis(query_embedding, top_k=chunk_size)
        print('search for redis ended')
    elif model == "chroma":
        print('search started for chroma')
        retrieved_results = search_chroma(query_embedding, top_k=chunk_size)
        print('search for chroma ended')
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
    results["model"] = model
    results["chunk_size"] = chunk_size
    results["embedding_time"] = round(embedding_time, 4)
    results["search_time"] = round(search_time, 4)
    results["response_time"] = round(response_time, 4)
    results["memory_usage"] = round(memory_after - memory_before, 2)
    results["retrieved_results"] = retrieved_results  # Store actual retrievals
    results["model_response"] = model_response  # Store model's generated respons

    return results



# Define different chunk sizes and models
top_k_valies = [1,2,5,10]
models = ["redis", "chroma"]
query = "Explain quantum computing"

# Store results in a list
benchmark_results = []
chunk_sizes = [200, 300, 500]
overlap = 50


for chunk_size in chunk_sizes:
    # Re-run ingestion for each chunk size
    print(f"\n--- Ingesting PDFs with chunk_size={chunk_size}, overlap={overlap} ---")
    clear_redis_store()  # Clear existing embeddings
    create_hnsw_index()
    process_pdfs(data_path, chunk_size=chunk_size, overlap=overlap)  # Pass new chunking parameters


for model in models:
    for k in top_k_valies:
        result = benchmark_search(query, model=model, chunk_size=k)
        #answer = generate_rag_response(query, search_embeddings(query), vector_model = model)
        benchmark_results.append(result)
        #benchmark_results.append(answer)

# Convert to DataFrame for easy analysis
df_results = pd.DataFrame(benchmark_results)
print(df_results)

# Save results to CSV for later use
df_results.to_csv("benchmark_results.csv", index=False)