import faiss
import redis
import chromadb
import pickle
import numpy as np
import ollama
from redis.commands.search.query import Query


# Load FAISS index
VECTOR_DIM = 768
faiss_index = faiss.read_index("faiss_index.bin")  # Ensure the index is in the right location
with open("faiss_metadata.pkl", "rb") as f:
    faiss_metadata = pickle.load(f)

# Initialize Redis client (make sure it's running before starting)
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

# Initialize Chroma client
client = chromadb.Client()
chroma_collection = client.get_or_create_collection("embedding_index")

# Function to get embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Function to compare the embeddings
def compare_embeddings(query_embedding, top_k=3):
    # Compare with FAISS
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(query_vector, top_k)

    # Collect top results from FAISS
    faiss_results = [
        {"file": faiss_metadata[indices[0][i]]["file"],
         "page": faiss_metadata[indices[0][i]]["page"],
         "chunk": faiss_metadata[indices[0][i]]["chunk"],
         "similarity": distances[0][i]}
        for i in range(len(indices[0])) if indices[0][i] != -1
    ]

    # Search Redis
    query_vector_bytes = query_vector.tobytes()
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "file", "page", "chunk", "vector_distance")
        .dialect(2)
    )
    redis_results = redis_client.ft("embedding_index").search(
        q, query_params={"vec": query_vector_bytes}
    )
    redis_results = [
        {"file": result.file, "page": result.page, "chunk": result.chunk, "similarity": result.vector_distance}
        for result in redis_results.docs
    ][:top_k]

    # Search ChromaDB
    chroma_results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chroma_results = [
        {
            "file": chroma_results["metadatas"][0][i]["file"],
            "page": chroma_results["metadatas"][0][i]["page"],
            "chunk": chroma_results["metadatas"][0][i]["chunk"],
            "similarity": chroma_results["distances"][0][i],
        }
        for i in range(len(chroma_results["ids"][0]))
    ]

    return {"faiss": faiss_results, "redis": redis_results, "chroma": chroma_results}

def interactive_compare():
    print(" Compare Embeddings Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Get embedding for the query
        query_embedding = get_embedding(query)

        # Compare embeddings from FAISS, Redis, and Chroma
        results = compare_embeddings(query_embedding)

        # Print the comparison results
        print("\n--- FAISS Results ---")
        for result in results["faiss"]:
            print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']}")

        print("\n--- Redis Results ---")
        for result in results["redis"]:
            print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']}")

        print("\n--- Chroma Results ---")
        for result in results["chroma"]:
            print(f"File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']}")

if __name__ == "__main__":
    interactive_compare()
