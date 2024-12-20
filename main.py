import os
import re
from typing import Optional

import chromadb
import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings

from config import config
from utils import logger


class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Optional[Embeddings]:
        try:
            response = ollama.embed(input=texts, model=config.task.embedding_model)
            return response.embeddings
        except Exception as e:
            logger.error(e)

        return None


class RAGSystem:
    def __init__(self):
        # self.chroma_client = chromadb.PersistentClient(path=config.paths.storage)
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self._setup_collection()

    @staticmethod
    def _get_nodes():
        from utils import list_files_in_folder
        paths = list_files_in_folder(config.paths.roots.data)

        from llama_index.core import Document
        documents: [Document] = []

        for file_path in paths:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            text = re.sub(r"\r", " ", text)
            text = re.sub(r"\s+", " ", text)
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
        from llama_index.core.node_parser import TokenTextSplitter

        pipeline = IngestionPipeline(transformations=[
                TokenTextSplitter(
                    chunk_size=config.task.chunk_size,
                    chunk_overlap=20,
                )
            ])

        nodes = pipeline.run(documents=documents)

        return nodes

    def _setup_collection(self):
        nodes = self._get_nodes()

        collection = self.chroma_client.get_or_create_collection(
            name='books',
            embedding_function=MyEmbeddingFunction()
        )

        for node in nodes:
            collection.add(
                ids=[node.node_id],
                documents=[node.text],
                metadatas=[{
                    'file_name': node.metadata['file_name'],
                    'file_type': node.metadata['file_type'],
                }],
            )

        return collection

    @staticmethod
    def _get_preferences():
        preferences = f"""
        - Answer in simple language.
        - Only use context information. Nothing else.
        - Answer the Question in 100 words or less.
        """

        return preferences

    def _get_context(self, prompt):
        context_results = self.collection.query(query_texts=[prompt], n_results=config.task.n_vector_results)
        texts = context_results["documents"][0]
        context = "\n".join(texts)
        context = context[:min(3500, len(context))]
        # context = context[:min(7500, len(context))]

        if config.task.eda_mode:
            context_samples = "\n---\n".join([text[:min(100, len(text))] for text in context_results["documents"][0]])
            logger.info('Context')
            print(context_samples)

        return context

    @staticmethod
    def _query_llm(context, user_prompt, preferences):
        prompt = f"""
        Context: {context}
        Preferences: {preferences}
        Question: {user_prompt}
        Answer: 
        """

        print(prompt)
        logger.info(f'Prompt Length : {len(prompt)}')

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt
        )

        return response

    def query(self, prompt):
        context = self._get_context(prompt)
        preferences = self._get_preferences()

        response = self._query_llm(context, prompt, preferences)
        response = response['response']
        return response


if __name__ == "__main__":
    rag_system = RAGSystem()

    user_prompt = "Who was the criminal in the Study in Scarlet?"
    logger.info('User Prompt')
    print(user_prompt)

    response = rag_system.query(user_prompt)

    logger.info('Response')
    print(response)
