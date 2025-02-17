# Retrieval-Augmented Generation

This is a retrieval-augmented generation (RAG) system that enhances language model responses by retrieving relevant information from a local knowledge base. It integrates BM25-based keyword search, semantic search using ChromaDB, and reranking with a cross-encoder to provide accurate and contextually relevant answers.

## Overview

This RAG system consists of the following components,

- **ChromaDB Vector Search** : Retrieves semantically similar documents from a Chroma collection
- **BM25 Lexical Search** : Extracts keyword-based matches using BM25 scoring
- **Hybrid Retrieval**: Merges search results from Chroma and BM25 using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Utilizes a transformer-based cross-encoder model `ms-marco-MiniLM-L-6-v2` to refine retrieved results.
- **Integration with SmolAgents**: Uses `smolagents` to facilitate interaction with the retrieval system.
- **Gradio Web Interface**: A simple UI for querying the system.

By default, the `Qwen 2.5` model is used for response generation via Ollama. 


## Usage

### Running 

To launch a web-based UI for querying the agent:

```sh
python main.py
```

- This will start a Gradio interface where you can enter queries.
- Open the generated Gradio UI URL in your browser
- Enter your query and get relevant retrieved snippets

### Code Example 

```python
from RAGAgent import RAGAgent

agent = RAGAgent()
response = agent.query("What is retrieval-augmented generation?")
print(response)
```

## Configuration

Modify `config.py` to adjust parameters such as:

- `n_results`: Number of retrieved documents
- `chunk_size`: Size of chunks taken from documents for indexing


## Evaluation

First set a local model as evaluation model.

```commandline
deepeval set-ollama deepseek-r1:8b
```
The `set DEEPEVAL_RESULTS_FOLDER=.\test_results`

Then run evaluate.

```commandline
deepeval test run .\tests\test_example.py
```

