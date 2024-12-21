import os
import re
from typing import List, Optional

import chromadb
import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

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


class SnippetRelevanceOutput(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class PromptTemplate:
    @classmethod
    def prepare_context(cls, snippets: List[str], empty_space: int) -> str:
        context = ''

        for i, text in enumerate(snippets):
            context += f"Context {i+1}:\n{text}\n"

        limit = min(empty_space, len(context))
        context = context[:limit]

        return context


class RAGSystem:
    def __init__(self):
        # self.chroma_client = chromadb.PersistentClient(path=config.paths.storage)
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self._setup_collection()
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @staticmethod
    def _get_nodes():
        from utils import list_files_in_folder
        paths = list_files_in_folder(config.paths.roots.data)

        from llama_index.core import Document
        documents: [Document] = []

        for file_path in paths:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            text = re.sub('\r', ' ', text)
            text = re.sub('\\s+', ' ', text)
            text = text.strip()

            file_name = os.path.basename(file_path)
            doc = Document(
                doc_id=file_name,
                text=text,
                extra_info={
                    'file_path': file_path,
                    'file_name': file_name,
                    'file_type': 'text/plain',
                }
            )

            documents.append(doc)

        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter

        pipeline = IngestionPipeline(transformations=[
                SentenceSplitter(
                    chunk_size=config.task.chunk_size,
                    chunk_overlap=50,
                )
            ])

        nodes = pipeline.run(documents=documents)

        return nodes

    def _setup_collection(self):
        # nodes hold the text snippets
        nodes = self._get_nodes()

        collection = self.chroma_client.get_or_create_collection(
            name='books',
            embedding_function=SnippetEmbeddingFunction()
        )

        # noinspection SpellCheckingInspection
        ids, documents, metadatas = [], [], []

        for node in nodes:
            ids.append(node.node_id)
            documents.append(node.text)
            metadatas.append({
                'file_name': node.metadata.get('file_name', 'unknown'),
                'file_type': node.metadata.get('file_type', 'unknown'),
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return collection

    @staticmethod
    def _query_llm(user_prompt, snippets):
        preferences = f"""
        - Answer in simple language.
        - Only use context information. Nothing else.
        - Answer the Question in 100 words or less.
        - Use lists when appropriate to break down complex information.
        """

        prompt = """Contexts: 
        {context}
        Preferences: 
        {preferences}
        Question: 
        {user_prompt}
        Answer: 
        """

        empty_space = config.llm[f'{config.task.llm}'].context_window_size
        empty_space = empty_space - len(preferences) - len(user_prompt) - len(prompt)

        context = PromptTemplate.prepare_context(snippets, empty_space)
        prompt = prompt.format(
            context=context,
            preferences=preferences,
            user_prompt=user_prompt
        )

        if config.task.debug_mode:
            print(prompt)
            logger.info(f'Prompt Length : {len(prompt)}')

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt
        )

        response = response['response']

        return response

    def _re_rank_cross_encoders(self, user_prompt, snippets):
        ranks = self.cross_encoder.rank(user_prompt, snippets)

        reranked_snippets = []
        for rank in ranks:
            snippet = snippets[rank['corpus_id']]
            reranked_snippets.append(snippet)

        return reranked_snippets

    @staticmethod
    def _rewrite_query(user_prompt):
        prompt = f"""You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
        
        Original query: {user_prompt}
        
        Rewritten query:
        """

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt
        )

        response = response['response']

        if config.task.debug_mode:
            logger.info('Rewritten User Prompt')
            print(response)

        return response

    def _get_snippets(self, user_prompt):
        results = self.collection.query(
            query_texts=user_prompt,
            n_results=config.task.n_vector_results
        )

        snippets = results["documents"][0]
        logger.info(f'Number of Snippets Retrieved : {len(snippets)}')

        if config.task.rerank:
            logger.info('Re-ranking Snippets')
            snippets = self._re_rank_cross_encoders(user_prompt, snippets)

        return snippets

    def query(self, user_prompt):
        if config.task.rewrite_user_prompt:
            user_prompt = self._rewrite_query(user_prompt)

        snippets = self._get_snippets(user_prompt)
        response = self._query_llm(user_prompt, snippets)

        return response


if __name__ == "__main__":
    rag_system = RAGSystem()

    user_prompt = "What is the address of Sherlock Holmes’ famous residence in London?"
    logger.info('User Prompt')
    print(user_prompt)

    response = rag_system.query(user_prompt)

    logger.info('Response')
    print(response)
