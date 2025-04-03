import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
import chromadb
import faiss
import pickle
from sentence_transformers import SentenceTransformer



ally = "/Users/alisonpicerno/Desktop/ds 4300/Dawg_Patrol_5_AI_Takeover/data"
connor = "/Users/connorgarmey/Documents/Large Scale/Practical 2/Dawg_Patrol_5_AI_Takeover/data"
data_path = ally


# Chunk settings
chunk_size = 300
overlap = 50

# Initialize Redis and Chroma
redis_client = redis.Redis(host="localhost", port=6380, db=0)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

#VECTOR_DIM = 768 #nomic
VECTOR_DIM = 1024 # embed-large
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
faiss_metadata = []


def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")
    
def clear_faiss_store():
    """Reset FAISS index and metadata"""
    global faiss_index, faiss_metadata
    faiss_index = faiss.IndexFlatL2(VECTOR_DIM)  # Recreate FAISS index
    faiss_metadata = []  # Reset metadata
    print("FAISS index cleared.")

def clear_chroma_store():
    try:
        existing_collections = chroma_client.list_collections()
        collection_names = [col.name for col in existing_collections]

        if f"{INDEX_NAME}_mistral" in collection_names:
            chroma_client.delete_collection(name=f"{INDEX_NAME}_mistral")
            print("Chroma collection deleted.")
        else:
            print("Chroma collection not found, skipping deletion.")

    except Exception as e:
        print(f"Error clearing Chroma store: {e}")

def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

def get_embedding(text, model_name):
    response = ollama.embeddings(model=model_name, prompt=text)
    return response["embedding"]

def store_embedding(file, page, chunk, embedding, model_name):
    model_name = model_name.replace(":", "_")
    key = f"{model_name}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )
    print(f"Stored embedding for: {chunk} under model: {model_name}")

def store_embedding_chroma(file, page, chunk, embedding, model_name):
    model_name = model_name.replace(":", "_")
    doc_id = f"{model_name}:{file}_page_{page}_chunk_{chunk}"
    chroma_collection = chroma_client.get_or_create_collection(
        name=f"{INDEX_NAME}_{model_name}", metadata={"hnsw:space": "cosine"}
    )
    chroma_collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    print(f"Stored embedding in ChromaDB for: {chunk} under model: {model_name}")

def store_embedding_faiss(file, page, chunk, embedding, model_name):
    global faiss_index, faiss_metadata
    model_name = model_name.replace(":", "_")
    vector = np.array(embedding, dtype=np.float32).reshape(1, -1)  # Ensure shape

    if vector.shape[1] != faiss_index.d:  # Check if dimensions match
        print(f"Dimension mismatch! Expected {faiss_index.d}, got {vector.shape[1]}")
        return
    
    faiss_index.add(vector)
    faiss_metadata.append({"file": file, "page": page, "chunk": chunk, "model": model_name})


def save_faiss_index():
    print('saving')
    """Save FAISS index and metadata."""
    faiss.write_index(faiss_index, "faiss_index.bin")
    with open("faiss_metadata.pkl", "wb") as f:
        pickle.dump(faiss_metadata, f)
    print("FAISS index and metadata saved.")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = [(page_num, page.get_text()) for page_num, page in enumerate(doc)]
    return text_by_page

# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size, overlap):
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Process all PDF files in a given directory
def process_pdfs(data_dir, chunk_size, overlap, models, embedding_model):
    global faiss_index, faiss_metadata
    print('models')
    if "faiss" in models:
        faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
        faiss_metadata = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                text = re.sub(r"[^\w\s.]", "", text)  # Remove punctuation except periods
                text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
                chunks = split_text_into_chunks(text, chunk_size, overlap)

                for chunk in chunks:
                    #for model_name, model_path in embedding_models.items():
                
                    embedding = get_embedding(chunk, embedding_model)
                    
                    if "redis" in models:
                        store_embedding(file_name, str(page_num), chunk, embedding, embedding_model)
                    if "chroma" in models:
                        store_embedding_chroma(file_name, str(page_num), chunk, embedding, embedding_model)
                    if "faiss" in models:
                        store_embedding_faiss(file_name, str(page_num), chunk, embedding, embedding_model)

            print(f"Processed {file_name}")
            print('\n')
            print('\n')
            print('\n')
            print('\n')
    
    save_faiss_index()

def main():
    clear_redis_store()
    create_hnsw_index()
    process_pdfs(data_path)
    print("\n---Done processing PDFs---\n")

if __name__ == "__main__":
    main()

