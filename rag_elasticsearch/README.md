# RAG Implementation with Elasticsearch Hybrid Search

This project implements a Retrieval-Augmented Generation (RAG) system using Elasticsearch as the vector database, featuring hybrid search capabilities that combine both semantic similarity and keyword matching.

## Features

- Document ingestion and vectorization using LlamaIndex
- Hybrid search combining vector similarity and keyword matching
- Configurable weighting between semantic and keyword search
- Support for various document formats through SimpleDirectoryReader
- Efficient vector storage and retrieval using Elasticsearch

## Prerequisites

- Python 3.8+
- Elasticsearch 8.x running locally or accessible via URL
- OpenAI API key (for LlamaIndex)

## Setup

1. Install Elasticsearch:
   - Download and install from [Elasticsearch official website](https://www.elastic.co/downloads/elasticsearch)
   - Start Elasticsearch service

2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
ELASTICSEARCH_URL=http://localhost:9200
```

## Usage

1. Place your documents in the `data` directory

2. Run the RAG system:
```bash
python rag_hybrid_search.py
```

## How It Works

### Document Ingestion
- Documents are loaded from the `data` directory
- Text is extracted and processed using LlamaIndex's SimpleDirectoryReader
- Documents are vectorized using the Sentence Transformers model
- Vectors and text are stored in Elasticsearch

### Hybrid Search
The system combines two types of search:
1. **Vector Similarity Search**: Uses cosine similarity between document and query embeddings
2. **Keyword Matching**: Uses Elasticsearch's standard text matching capabilities

The final score is computed as:
```
final_score = α * vector_score + (1-α) * keyword_score
```
where α is configurable (default: 0.5)

## Customization

You can customize the following parameters:
- `embedding_model`: Change the embedding model (default: 'sentence-transformers/all-MiniLM-L6-v2')
- `alpha`: Adjust the weight between vector and keyword search
- `k`: Number of results to return

## Example

```python
from rag_hybrid_search import ElasticsearchRAG

# Initialize RAG system
rag = ElasticsearchRAG()

# Ingest documents
rag.ingest_documents("data")

# Perform hybrid search
results = rag.hybrid_search(
    query="What are the main benefits of RAG systems?",
    k=5,
    alpha=0.7  # Favor vector similarity over keyword matching
)
```

## Contributing

Feel free to submit issues and enhancement requests! 