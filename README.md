Goal:
-----
Compare the performance of three popular vector databases (Redis, Chroma, and FAISS) for search over PDF documents using different embedding models and chunking strategies.

Installation:
-------------
1. Clone the repository

2. Install dependencies:
- pip install chromadb
- pip install redis
- pip install faiss-cpu
- pip install psutil

4. Start Redis-stack server if using Redis as a vector store

5. Start Ollama


How to Use:
-----------
1. Place your PDFs in the `data/` folder.
2. Run the benchmarking script:
   python src/compare.py

You can customize the script to test:
- Different chunk sizes
- Overlap values
- Top-k retrieval values
- Embedding models
- Vector stores

Benchmark results will be saved in a CSV file: `benchmark_results.csv`.

Output:
-------
The results include the following fields:
- Chunk size and overlap
- Vector store used
- Top-k retrieval size
- Embedding, search, and response generation times
- Memory usage
- Generated model response

Embedding Models to Test:
-----------------
- nomic-embed-text:latest
- mxbai-embed-large
- snowflake-arctic-embed
