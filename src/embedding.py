from typing import Optional
from chromadb import Documents, EmbeddingFunction, Embeddings

import ollama

from config import config
from utils import logger


class SnippetEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Optional[Embeddings]:
        try:
            response = ollama.embed(input=texts, model=config.task.embedding_model)
            return response.embeddings
        except Exception as e:
            logger.error(e)

        return None

