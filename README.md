# Retrieval-Augmented Generation

## Overview

This RAG system consists of the following components,

1. Query Reformulation 
   - Optionally, using an LLM we can refine user prompts to optimize retrieval quality without altering intent.
2. ChromaDB Vector Search  
   - Retrieves semantically similar documents from a Chroma collection
3. BM25 Lexical Search 
   - Extracts keyword-based matches using BM25 scoring
4. Hybrid Results Fusion
   - Combines Chroma and BM25 results using Reciprocal Rank Fusion (RRF)
5. Re-ranking
   - A pre-trained CrossEncoder model is used to re-rank these results

Finally, we put together the user's preferences and question to generate responses. A `Phi 3.5`
model is used for response generation via Ollama. 


```python
rag = RAGSystem()
response = rag.query("What is the capital of France?")
print(response)
```

## Evaluation

First set a local model as evaluation model.

```commandline
deepeval set-local-model --model-name=gemma2:9b --base-url="http://localhost:11434/v1/"
```

Then run evaluate.

```commandline
python .\evaluate.py
```
