from config import config
from utils import logger

from typing import Optional
from chromadb import Documents, EmbeddingFunction, Embeddings

from sentence_transformers import SentenceTransformer
import ollama


# class SnippetEmbeddingFunction(EmbeddingFunction):
#     def __call__(self, texts: Documents) -> Optional[Embeddings]:
#         try:
#             response = ollama.embed(input=texts, model=config.task.embedding_model)
#             return response.embeddings
#         except Exception as e:
#             logger.error(e)
#
#         return None


class SnippetEmbeddingFunction(EmbeddingFunction):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def __call__(self, texts: Documents):
        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(e)

        return None
