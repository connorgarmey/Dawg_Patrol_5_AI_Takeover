
Installation:
-------------
1. Clone the repository

2. Install dependencies:
  pip install chromadb
  pip install redis
  pip install faiss-cpu
  pip install psutil

3. Start Redis-stack server if using Redis as a vector store:

4. Start Ollama


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

Embedding Models:
-----------------
- nomic-embed-text: Converts text into high-dimensional vectors for semantic search and ML tasks.
- mxbai-embed-large: General-purpose embedding model trained for multilingual and domain-agnostic applications.
- snowflake-arctic-embed: Embedding model optimised for enterprise and tabular text tasks, developed by Snowflake.

