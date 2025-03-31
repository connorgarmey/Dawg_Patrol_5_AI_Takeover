import redis
import json
import numpy as np
#from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import chromadb

vector_model = 'redis'

# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)
# Initialize the Chroma client
client = chromadb.Client()

# Access our embedding_index collection
chroma_collection = client.get_or_create_collection("embedding_index")

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3, vector_model=vector_model):
    # Search embeddings using specified model
    query_embedding = get_embedding(query)

    if vector_model == "redis":
        return search_redis(query_embedding, top_k)
    elif vector_model == "chroma":
        return search_chroma(query_embedding, top_k)
    else:
        print("Invalid model specified:", vector_model)
        return 
    
def search_redis(query_embedding, top_k):
    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

# Searching ChromaDB
def search_chroma(query_embedding, top_k):
    """Search for similar embeddings in ChromaDB."""
    try:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        top_results = [
            {
                "file": results["metadatas"][0][i]["file"],
                "page": results["metadatas"][0][i]["page"],
                "chunk": results["metadatas"][0][i]["chunk"],
                "similarity": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

        print("\n--- ChromaDB Search Results ---")
        for result in top_results:
            print(f"File: {result['file']}, Page: {result['page']}, Similarity: {result['similarity']}")

        return top_results

    except Exception as e:
        print(f"ChromaDB search error: {e}")
        return []



def generate_rag_response(query, context_results):

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

   # print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
# 3 highest chunks returned from vector db
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama, this is where you change the model
    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    #Chat GPT like
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )
if __name__ == "__main__":
    interactive_search()
