import os
import re
from typing import Optional

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


class RAGSystem:
    def __init__(self):
        # self.chroma_client = chromadb.PersistentClient(path=config.paths.storage)
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self._setup_collection()

    @staticmethod
    def _show_context(contexts):
        for i, text in enumerate(contexts):
            print(f"Context {i+1}:")
            print(text[:min(100, len(text))])

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
        nodes = self._get_nodes()  # nodes hold the text snippets

        collection = self.chroma_client.get_or_create_collection(
            name='books',
            embedding_function=SnippetEmbeddingFunction()
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
    def _is_snippet_relevant(user_prompt, snippet):
        prompt = f"""
        You are a grader assessing relevance of a snippet to a user question.
        - If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        - Do not be too strict. The goal is just to filter out extremely erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        
        Snippet: {snippet}
        Question: {user_prompt}
        """

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt,
            format=SnippetRelevanceOutput.model_json_schema(),
        )

        relevance = SnippetRelevanceOutput.model_validate_json(response.response)
        relevance = relevance.binary_score.strip().lower()

        if relevance not in ['yes', 'no']:
            logger.debug(f'Invalid Relevance Score: {relevance}')

        relevance = relevance == 'yes'
        return relevance

    @staticmethod
    def re_rank_cross_encoders(user_prompt, snippets):
        encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranks = encoder_model.rank(user_prompt, snippets)

        reranked_snippets = []

        for rank in ranks:
            snippet = snippets[rank['corpus_id']]
            reranked_snippets.append(snippet)

        return reranked_snippets

    @staticmethod
    def _truncate_context(context):
        limit = config.llm[config.task.llm].context_limit
        if len(context) > limit:
            logger.warning(f"Context truncated to {limit} characters.")

        limit = min(config.llm[config.task.llm].context_limit, len(context))
        return context[:limit]

    def _get_context(self, user_prompt):
        results = self.collection.query(
            query_texts=[user_prompt],
            n_results=config.task.n_vector_results
        )

        snippets = results["documents"][0]
        logger.info(f'Number of Contexts Retrieved : {len(snippets)}')

        if config.task.check_relevance:  # too binary
            filtered_snippets = [s for s in snippets if self._is_snippet_relevant(user_prompt, s)]
            logger.info(f'Number of Relevant Contexts : {len(filtered_snippets)}')

            if len(filtered_snippets) == 0:
                logger.warning('All contexts are deemed irrelevant. Using all contexts instead.')
                snippets = filtered_snippets

        if config.task.rerank:
            logger.info('Re-ranking Contexts')
            snippets = self.re_rank_cross_encoders(user_prompt, snippets)

        context = "\n---\n".join(snippets)  # give proper sections
        context = self._truncate_context(context)

        if config.task.eda_mode:
            logger.info('Contexts')
            self._show_context(context)

        return context

    @staticmethod
    def _query_llm(context, user_prompt):
        preferences = f"""
        - Answer in simple language.
        - Only use context information. Nothing else.
        - Answer the Question in 100 words or less.
        - Use lists when appropriate to break down complex information.
        """

        prompt = f"""
        Context: {context}
        Preferences: {preferences}
        Question: {user_prompt}
        Answer: 
        """

        if config.task.eda_mode:
            print(prompt)

        logger.info(f'Prompt Length : {len(prompt)}')

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt
        )

        response = response['response']

        return response

    @staticmethod
    def _rewrite_query(user_prompt):
        prompt = f"""
        You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
        
        Original query: {user_prompt}
        
        Rewritten query:
        """

        response = ollama.generate(
            model=config.task.llm,
            prompt=prompt
        )

        response = response['response']

        return response

    def query(self, user_prompt):
        if config.task.rewrite_user_prompt:
            user_prompt = self._rewrite_query(user_prompt)
            logger.info('Rewritten User Prompt')
            print(user_prompt)

        context = self._get_context(user_prompt)

        response = self._query_llm(context, user_prompt)
        return response


if __name__ == "__main__":
    rag_system = RAGSystem()

    user_prompt = "What is the address of Sherlock Holmes’ famous residence in London?"
    logger.info('User Prompt')
    print(user_prompt)

    response = rag_system.query(user_prompt)

    logger.info('Response')
    print(response)
