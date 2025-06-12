# Retrieval-Augmented Generation

This project implements a Retrieval-Augmented Generation (RAG) workflow that uses,

* 🕸️ Web search via **Tavily**
* 📚 Document retrieval via **vector database**
* 💬 Local LLMs via **Ollama**
* 🔀 Dynamic reasoning and control flow using **LangGraph**
* 📊 Tracing and evaluation using **Langfuse**


## 🚀 Features

* Hybrid RAG with both web and vector-based retrieval
* Automatic query reformulation
* Relevance filtering before answer generation
* Hallucination detection and correction
* Tracing via Langfuse
* Modular node/edge logic using LangGraph

This RAG system is built with a hybrid retrieval architecture combining multiple search strategies:

* **ChromaDB Vector Search**: Retrieves documents based on semantic similarity using dense embeddings from a Chroma collection.
* **BM25 Lexical Search**: Identifies relevant documents using traditional keyword-based matching with BM25 scoring.
* **Hybrid Retrieval**: Integrates results from both semantic and lexical searches using **Reciprocal Rank Fusion (RRF)** for balanced and robust retrieval.


## 🔐 Environment Variables

Create a `.env` file in the project root:

```dotenv
TAVILY_API_KEY=your_tavily_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```
